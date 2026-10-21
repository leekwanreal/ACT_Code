#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Resampling and Timestamp Synchronization Module for OpenArm ACT
Resolves Bug 1: Frequency mismatch, variable dt jitter, and dropped frames.

Key capabilities:
1. Robust timestamp discovery across multiple HDF5 naming conventions
2. Temporal alignment onto a strictly uniform 50.0 Hz grid (dt = 0.020s)
3. Trajectory interpolation for continuous signals (qpos, action, qvel, effort)
4. Time-synchronized nearest-frame selection for RGB-D camera feeds
5. Offline HDF5 dataset resampling tool and in-memory dataloader caching
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import h5py


# Candidate keys for timestamps in HDF5 hierarchy
TIMESTAMP_CANDIDATE_KEYS = [
    "timestamp_ns",
    "timestamps_ns",
    "timestamps",
    "timestamp",
    "time",
    "observations/timestamp_ns",
    "observations/timestamps_ns",
    "observations/timestamps",
    "observations/timestamp",
    "observations/time",
    "images/timestamp_ns",
    "images/timestamps",
    "observations/images/timestamp_ns",
    "observations/images/timestamps",
]


def extract_timestamps(
    h5_file: h5py.File,
    num_steps: int,
    default_hz: float = 50.0,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Trích xuất và chuẩn hóa vector thời gian (đơn vị: giây, t[0] = 0.0) từ file HDF5.

    Args:
        h5_file: Đối tượng h5py.File đã mở
        num_steps: Số lượng bước timesteps (T) của dữ liệu góc khớp/hành động
        default_hz: Tần số danh định (mặc định 50.0 Hz)

    Returns:
        t: np.ndarray shape [T], kiểu float64, tính bằng giây, bắt đầu từ 0.0
        has_explicit_ts: bool, True nếu tìm thấy timestamps thực tế
        stats: dict chứa thông tin chuẩn đoán (actual_hz, min_dt, max_dt, jitter_detected, ...)
    """
    raw_ts = None
    found_key = None

    # 1. Tìm dataset timestamps theo danh sách key thông dụng
    for key in TIMESTAMP_CANDIDATE_KEYS:
        if key in h5_file:
            raw_data = h5_file[key]
            if len(raw_data) == num_steps and len(raw_data) >= 2:
                raw_ts = np.array(raw_data[:], dtype=np.float64)
                found_key = key
                break

    # 2. Nếu tìm thấy dataset timestamps
    if raw_ts is not None:
        has_explicit_ts = True
        # Xác định đơn vị: ns, us, ms hay seconds dựa trên bước nhảy trung bình
        diffs = np.diff(raw_ts)
        valid_diffs = diffs[diffs > 0]
        median_diff = np.median(valid_diffs) if len(valid_diffs) > 0 else 1.0

        if median_diff > 1e6:       # Nanoseconds (ví dụ ~20,000,000 ns = 20ms)
            scale = 1e-9
        elif median_diff > 1e3:     # Microseconds
            scale = 1e-6
        elif median_diff > 0.5 and raw_ts[0] > 1e6: # Milliseconds
            scale = 1e-3
        else:                       # Seconds
            scale = 1.0

        t = (raw_ts - raw_ts[0]) * scale
        # Đảm bảo tính đơn điệu không giảm (tránh lỗi xung đột do đồng hồ hệ thống giật lùi)
        t = np.maximum.accumulate(t)

    else:
        # 3. Kiểm tra thuộc tính metadata (Attributes) nếu có duration_seconds
        has_explicit_ts = False
        duration_sec = None

        for attr_target in [h5_file.attrs, h5_file.get("observations", {}).attrs if "observations" in h5_file else {}]:
            for attr_name in ["duration_seconds", "duration_sec", "duration", "total_time"]:
                if attr_name in attr_target:
                    try:
                        duration_sec = float(attr_target[attr_name])
                        found_key = f"attrs[{attr_name}]"
                        has_explicit_ts = True
                        break
                    except (ValueError, TypeError):
                        pass
            if duration_sec is not None:
                break

        if duration_sec is not None and duration_sec > 0:
            t = np.linspace(0.0, duration_sec, num_steps, dtype=np.float64)
        else:
            # Fallback: Tự sinh timestamp danh định
            nominal_hz = float(h5_file.attrs.get("frequency_hz", default_hz))
            t = np.arange(num_steps, dtype=np.float64) / nominal_hz
            found_key = f"nominal_grid_{nominal_hz}hz"

    # 4. Tính toán các chỉ số chẩn đoán (Diagnostics)
    total_duration = float(t[-1]) if len(t) > 1 else 0.0
    actual_hz = float((num_steps - 1) / total_duration) if total_duration > 0 else default_hz

    dts = np.diff(t)
    min_dt = float(np.min(dts)) if len(dts) > 0 else 0.0
    max_dt = float(np.max(dts)) if len(dts) > 0 else 0.0
    median_dt = float(np.median(dts)) if len(dts) > 0 else 0.0
    std_dt = float(np.std(dts)) if len(dts) > 0 else 0.0

    nominal_dt = 1.0 / default_hz
    # Phát hiện jitter: Tần số lệch > 1.0 Hz HOẶC khoảng cách max_dt - min_dt > 5ms
    jitter_detected = (abs(actual_hz - default_hz) > 1.0) or (max_dt - min_dt > 0.005) or (std_dt > 0.003)

    # Ước tính số frame bị rớt (frame drop)
    expected_steps = int(round(total_duration * default_hz)) + 1
    dropped_frames = max(0, expected_steps - num_steps)
    drop_pct = (dropped_frames / max(expected_steps, 1)) * 100.0

    stats = {
        "found_key": found_key,
        "has_explicit_ts": has_explicit_ts,
        "num_steps": num_steps,
        "duration_seconds": round(total_duration, 4),
        "actual_hz": round(actual_hz, 2),
        "min_dt": round(min_dt, 5),
        "max_dt": round(max_dt, 5),
        "median_dt": round(median_dt, 5),
        "std_dt": round(std_dt, 5),
        "jitter_detected": jitter_detected,
        "expected_steps_at_50hz": expected_steps,
        "estimated_dropped_frames": dropped_frames,
        "drop_pct": round(drop_pct, 2),
    }

    return t, has_explicit_ts, stats


def compute_resampling_plan(
    t_raw: np.ndarray,
    target_hz: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xây dựng kế hoạch lấy mẫu lại đồng đều theo chuẩn target_hz (50Hz = dt=0.02s).

    Args:
        t_raw: Mảng timestamp gốc [T_raw] tính bằng giây (bắt đầu tại 0.0)
        target_hz: Tần số mục tiêu (50.0 Hz)

    Returns:
        t_uniform: Lưới thời gian đồng đều [T_new] = [0.0, 0.02, 0.04, ..., T_end]
        frame_indices: Mảng chỉ số frame ảnh gần nhất [T_new], ánh xạ mỗi bước thời gian
                       đồng đều về frame ảnh thực tế thu được từ camera.
    """
    total_duration = t_raw[-1]
    target_dt = 1.0 / target_hz

    # Tạo lưới thời gian đồng đều chính xác
    t_uniform = np.arange(0.0, total_duration, target_dt, dtype=np.float64)
    # Đảm bảo ít nhất có 1 bước và nếu điểm cuối gần với total_duration thì bao hàm
    if len(t_uniform) == 0 or (total_duration - t_uniform[-1]) >= (target_dt * 0.5):
        t_uniform = np.append(t_uniform, total_duration)

    # Tìm chỉ số frame ảnh gần nhất cho từng thời điểm t_uniform[k] (Nearest-neighbor)
    # Vectorized: searchsorted + so sánh khoảng cách với điểm bên trái và bên phải
    right_indices = np.searchsorted(t_raw, t_uniform)
    right_indices = np.clip(right_indices, 0, len(t_raw) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(t_raw) - 1)

    dist_left = np.abs(t_raw[left_indices] - t_uniform)
    dist_right = np.abs(t_raw[right_indices] - t_uniform)

    frame_indices = np.where(dist_left <= dist_right, left_indices, right_indices)

    return t_uniform, frame_indices


def resample_signals(
    t_raw: np.ndarray,
    signals: np.ndarray,
    t_uniform: np.ndarray,
) -> np.ndarray:
    """
    Nội suy các tín hiệu robot liên tục (qpos, action, qvel, effort) lên lưới thời gian đồng đều.
    Sử dụng nội suy tuyến tính (Linear Interpolation) chống hiện tượng dao động lố (overshoot)
    và đảm bảo luôn nằm trong giới hạn cơ học an toàn của các khớp.

    Args:
        t_raw: [T_raw]
        signals: [T_raw, D] hoặc [T_raw]
        t_uniform: [T_uniform]

    Returns:
        resampled: [T_uniform, D] hoặc [T_uniform] cùng shape
    """
    is_1d = (signals.ndim == 1)
    if is_1d:
        signals = signals[:, None]

    num_dims = signals.shape[1]
    resampled = np.empty((len(t_uniform), num_dims), dtype=np.float32)

    for d in range(num_dims):
        resampled[:, d] = np.interp(t_uniform, t_raw, signals[:, d])

    if is_1d:
        return resampled[:, 0]
    return resampled


def resample_hdf5_file(
    src_path: str,
    dst_path: str,
    target_hz: float = 50.0,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Dict[str, Any]:
    """
    Đọc một file HDF5 gốc, lấy mẫu lại toàn bộ dữ liệu về tần số chuẩn target_hz (50Hz),
    đồng bộ hóa thời gian và lưu thành file HDF5 mới hoàn toàn chuẩn hóa.

    Args:
        src_path: Đường dẫn file HDF5 nguồn
        dst_path: Đường dẫn file HDF5 đích
        target_hz: Tần số chuẩn (50.0 Hz)
        compression: Kiểu nén dữ liệu HDF5 ("gzip" hoặc None)
        compression_opts: Mức độ nén (1-9)

    Returns:
        report: Dict ghi nhận chi tiết số bước cũ/mới, tỷ lệ rớt khung hình, v.v.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)

    with h5py.File(src_path, "r") as f_src:
        T_raw = len(f_src["action"])

        # 1. Trích xuất timestamps và xây dựng kế hoạch lấy mẫu lại
        t_raw, has_explicit_ts, stats = extract_timestamps(f_src, T_raw, default_hz=target_hz)
        t_uniform, frame_indices = compute_resampling_plan(t_raw, target_hz=target_hz)
        T_new = len(t_uniform)

        # 2. Tạo file HDF5 mới
        with h5py.File(dst_path, "w") as f_dst:
            # Sao chép và cập nhật Metadata Attributes
            for k, v in f_src.attrs.items():
                f_dst.attrs[k] = v
            f_dst.attrs["frequency_hz"] = float(target_hz)
            f_dst.attrs["resampled_to_hz"] = float(target_hz)
            f_dst.attrs["original_samples"] = int(T_raw)
            f_dst.attrs["resampled_samples"] = int(T_new)
            f_dst.attrs["duration_seconds"] = float(t_raw[-1])
            f_dst.attrs["resampled_from"] = os.path.basename(src_path)

            # Ghi timestamp chuẩn đồng đều mới (nanoseconds và seconds)
            f_dst.create_dataset(
                "timestamp_ns",
                data=(t_uniform * 1e9).astype(np.int64),
                compression=compression,
            )

            # 3. Nội suy các vector liên tục: qpos, action, qvel, effort
            obs_src = f_src["observations"]
            obs_dst = f_dst.create_group("observations")
            for k, v in obs_src.attrs.items():
                obs_dst.attrs[k] = v

            # qpos
            if "qpos" in obs_src:
                raw_qpos = obs_src["qpos"][:]
                new_qpos = resample_signals(t_raw, raw_qpos, t_uniform)
                obs_dst.create_dataset("qpos", data=new_qpos, dtype=np.float32, compression=compression)

            # action
            if "action" in f_src:
                raw_action = f_src["action"][:]
                new_action = resample_signals(t_raw, raw_action, t_uniform)
                f_dst.create_dataset("action", data=new_action, dtype=np.float32, compression=compression)

            # qvel (nếu có)
            if "qvel" in obs_src:
                raw_qvel = obs_src["qvel"][:]
                new_qvel = resample_signals(t_raw, raw_qvel, t_uniform)
                obs_dst.create_dataset("qvel", data=new_qvel, dtype=np.float32, compression=compression)

            # effort (nếu có)
            if "effort" in obs_src:
                raw_effort = obs_src["effort"][:]
                new_effort = resample_signals(t_raw, raw_effort, t_uniform)
                obs_dst.create_dataset("effort", data=new_effort, dtype=np.float32, compression=compression)

            # 4. Đồng bộ ảnh camera RGB-D bằng Nearest-Frame
            if "images" in obs_src:
                img_src_grp = obs_src["images"]
                img_dst_grp = obs_dst.create_group("images")
                for k, v in img_src_grp.attrs.items():
                    img_dst_grp.attrs[k] = v

                for img_key in img_src_grp.keys():
                    src_ds = img_src_grp[img_key]
                    # Tạo dataset đích với chiều T_new
                    new_shape = (T_new,) + src_ds.shape[1:]
                    dst_ds = img_dst_grp.create_dataset(
                        img_key,
                        shape=new_shape,
                        dtype=src_ds.dtype,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

                    # Ghi từng frame theo frame_indices để tối ưu RAM
                    # Gom các frame trùng lặp để giảm I/O
                    for i_new, i_raw in enumerate(frame_indices):
                        dst_ds[i_new] = src_ds[i_raw]

    report = {
        "src_file": os.path.basename(src_path),
        "dst_file": os.path.basename(dst_path),
        "original_samples": T_raw,
        "resampled_samples": T_new,
        "duration_seconds": stats["duration_seconds"],
        "actual_hz": stats["actual_hz"],
        "target_hz": target_hz,
        "drop_pct": stats["drop_pct"],
        "jitter_detected": stats["jitter_detected"],
    }
    return report

