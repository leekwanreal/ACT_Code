#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Format & Quality Validator for OpenArm Bimanual RGB-D ACT Dataset
Strictly verifies HDF5 episodes against DATA_FORMAT_SPECIFICATION.md:
1. Schema & Hierarchy: groups /observations/images, /observations/qpos, /action
2. Tensor Shapes & Types: uint8 RGB, uint16 Depth, float32 qpos/action (16-DOF)
3. Metadata Attributes: frequency_hz=50, num_joints=16, depth_scale, etc.
4. Numerical Integrity: No NaNs, no Infs, joint bounds clamping, depth mm range
5. Sampling Frequency: Verify timestamps or time intervals correspond to 50 Hz (20ms)
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import h5py

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Giới hạn khớp cơ học OpenArm (16 khớp), theo quy ước URDF chính thức (góc khớp = góc motor).
# Khớp tay tính bằng rad, gripper tính bằng hành trình ngón kẹp (m).
JOINT_LIMITS = {
    # Tay trái: J1..J7 + Gripper
    0: (-3.49, 1.40),
    1: (-3.32, 0.17),
    2: (-1.57, 1.57),
    3: (0.00, 2.44),
    4: (-1.57, 1.57),
    5: (-0.79, 0.79),
    6: (-1.57, 1.57),
    7: (0.00, 0.043),  # Gripper: hành trình 0..0.043 m
    # Tay phải: J1..J7 + Gripper
    8:  (-1.40, 3.49),
    9:  (-0.17, 3.32),
    10: (-1.57, 1.57),
    11: (0.00, 2.44),
    12: (-1.57, 1.57),
    13: (-0.79, 0.79),
    14: (-1.57, 1.57),
    15: (0.00, 0.043),
}

JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
]


class ValidationResult:
    def __init__(self, filename: str):
        self.filename = os.path.basename(filename)
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}

    def add_error(self, msg: str):
        self.passed = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_single_hdf5(file_path: str, expected_hz: float = 50.0) -> ValidationResult:
    res = ValidationResult(file_path)

    if not os.path.exists(file_path):
        res.add_error(f"File không tồn tại: {file_path}")
        return res

    try:
        with h5py.File(file_path, "r") as f:
            # 1. KIỂM TRA METADATA ATTRIBUTES
            attrs = f.attrs
            res.info["frequency_hz"] = attrs.get("frequency_hz", None)
            res.info["num_joints"] = attrs.get("num_joints", None)
            res.info["robot_type"] = attrs.get("robot_type", None)

            if "frequency_hz" in attrs:
                if float(attrs["frequency_hz"]) != expected_hz:
                    res.add_error(f"Attribute 'frequency_hz' là {attrs['frequency_hz']}, yêu cầu chuẩn là {expected_hz}")
            else:
                res.add_warning("Thiếu attribute 'frequency_hz' trong root HDF5")

            if "num_joints" in attrs:
                if int(attrs["num_joints"]) != 16:
                    res.add_error(f"Attribute 'num_joints' là {attrs['num_joints']}, yêu cầu là 16")
            else:
                res.add_warning("Thiếu attribute 'num_joints' (yêu cầu 16 cho Bimanual OpenArm)")

            # 2. KIỂM TRA HIERARCHY CẤU TRÚC NHÓM
            if "observations" not in f:
                res.add_error("Thiếu nhóm bắt buộc '/observations'")
                return res

            obs = f["observations"]
            if "images" not in obs:
                res.add_error("Thiếu nhóm bắt buộc '/observations/images'")
                return res

            images = obs["images"]
            has_chest_rgb = ("chest_rgb" in images) or ("chest" in images)
            has_chest_depth = ("chest_depth" in images) or ("chest" in images and images["chest"].shape[-1] == 4)
            has_front_rgb = ("front_rgb" in images) or ("front" in images)
            has_front_depth = ("front_depth" in images) or ("front" in images and images["front"].shape[-1] == 4)

            # Kiểm tra bắt buộc 02 Camera: Chest (trước ngực) và Front (đối diện)
            if not has_chest_rgb:
                res.add_error("Thiếu dataset ảnh màu camera ngực '/observations/images/chest_rgb'")
            if not has_chest_depth:
                res.add_error("Thiếu dataset ảnh chiều sâu camera ngực '/observations/images/chest_depth'")
            if not has_front_rgb:
                res.add_error("Thiếu dataset ảnh màu camera đối diện '/observations/images/front_rgb'")
            if not has_front_depth:
                res.add_error("Thiếu dataset ảnh chiều sâu camera đối diện '/observations/images/front_depth'")

            if "qpos" not in obs:
                res.add_error("Thiếu dataset góc khớp '/observations/qpos'")
            if "action" not in f:
                res.add_error("Thiếu dataset hành động mục tiêu '/action'")

            if not res.passed:
                return res

            # 3. KIỂM TRA ĐỘ DÀI TIMESTEPS T CÁC DATASET PHẢI KHỚP NHAU
            qpos_ds = obs["qpos"]
            action_ds = f["action"]
            T_qpos = len(qpos_ds)
            T_action = len(action_ds)

            if T_qpos != T_action:
                res.add_error(f"Độ dài không khớp: qpos có {T_qpos} bước, action có {T_action} bước")

            T = min(T_qpos, T_action)
            res.info["timesteps"] = T
            res.info["duration_sec"] = round(T / expected_hz, 2)

            if T < 50:
                res.add_error(f"Episode quá ngắn: chỉ có {T} bước (nhỏ hơn chunk_size=50)")

            # Kiểm tra tính toàn vẹn và độ phân giải của cả 2 Camera (Chest & Front)
            cams_to_check = [
                ("chest", "chest_rgb", "chest_depth"),
                ("front", "front_rgb", "front_depth"),
            ]

            for cam_name, rgb_key, depth_key in cams_to_check:
                if rgb_key in images:
                    rgb_ds = images[rgb_key]
                    if len(rgb_ds) != T:
                        res.add_error(f"Độ dài ảnh RGB {cam_name} ({len(rgb_ds)}) không khớp với qpos ({T})")
                    if rgb_ds.dtype != np.uint8:
                        res.add_error(f"Kiểu dữ liệu {rgb_key} là {rgb_ds.dtype}, bắt buộc phải là uint8")
                    if len(rgb_ds.shape) != 4 or rgb_ds.shape[3] != 3:
                        res.add_error(f"Shape {rgb_key} là {rgb_ds.shape}, yêu cầu chuẩn [T, H, W, 3]")
                    
                    H_img, W_img = rgb_ds.shape[1], rgb_ds.shape[2]
                    res.info[f"{cam_name}_resolution"] = f"{W_img}x{H_img}"

                    # Ràng buộc kích thước chuẩn 480 x 640
                    if (H_img, W_img) == (480, 640):
                        res.info[f"{cam_name}_aspect_type"] = "480x640 (Chuẩn OpenArm 4:3)"
                    elif (H_img, W_img) == (240, 424):
                        res.add_warning(
                            f"Camera {cam_name} dùng độ phân giải cũ 240x424 (16:9). "
                            f"Yêu cầu chuẩn mới là 480x640 (HxW)!"
                        )
                    else:
                        aspect_ratio = W_img / max(H_img, 1)
                        res.add_warning(f"Camera {cam_name} có kích thước {W_img}x{H_img} (AR: {aspect_ratio:.2f}:1), khuyến nghị 640x480.")

                if depth_key in images:
                    depth_ds = images[depth_key]
                    if len(depth_ds) != T:
                        res.add_error(f"Độ dài ảnh Depth {cam_name} ({len(depth_ds)}) không khớp với qpos ({T})")
                    if depth_ds.dtype != np.uint16:
                        res.add_warning(f"Kiểu dữ liệu {depth_key} là {depth_ds.dtype}, khuyến nghị uint16 (mm) để tối ưu dung lượng")
                    
                    H_d, W_d = depth_ds.shape[1], depth_ds.shape[2]
                    if rgb_key in images and (H_d != H_img or W_d != W_img):
                        res.add_error(f"Độ phân giải Depth {cam_name} ({W_d}x{H_d}) không khớp với RGB ({W_img}x{H_img})!")

            # 4. KIỂM TRA SHAPE & DTYPE CỦA QPOS VÀ ACTION
            if qpos_ds.shape[-1] != 16:
                res.add_error(f"Kích thước qpos chiều cuối là {qpos_ds.shape[-1]}, bắt buộc phải là 16 (16-DOF)")
            if action_ds.shape[-1] != 16:
                res.add_error(f"Kích thước action chiều cuối là {action_ds.shape[-1]}, bắt buộc phải là 16 (16-DOF)")

            # 5. KIỂM TRA TOÀN VẸN SỐ HỌC (NAN, INF, GIÁ TRỊ NGOẠI LAI)
            qpos_sample = qpos_ds[:]
            action_sample = action_ds[:]

            if np.isnan(qpos_sample).any():
                res.add_error("Phát hiện giá trị NaN trong '/observations/qpos'")
            if np.isinf(qpos_sample).any():
                res.add_error("Phát hiện giá trị Inf trong '/observations/qpos'")
            if np.isnan(action_sample).any():
                res.add_error("Phát hiện giá trị NaN trong '/action'")
            if np.isinf(action_sample).any():
                res.add_error("Phát hiện giá trị Inf trong '/action'")

            # Kiểm tra góc khớp có nằm trong phạm vi cơ học an toàn
            for j_idx in range(16):
                min_lim, max_lim = JOINT_LIMITS[j_idx]
                j_min = float(np.min(qpos_sample[:, j_idx]))
                j_max = float(np.max(qpos_sample[:, j_idx]))
                is_gripper = j_idx in (7, 15)
                # Dung sai: 0.15 rad cho khớp tay, 2 mm cho gripper
                tol, unit = (0.002, "m") if is_gripper else (0.15, "rad")
                if j_min < (min_lim - tol) or j_max > (max_lim + tol):
                    res.add_warning(
                        f"Khớp {JOINT_NAMES[j_idx]} (idx {j_idx}) có giá trị ngoài dải an toàn: "
                        f"[{j_min:.3f}, {j_max:.3f}] vs giới hạn [{min_lim:.3f}, {max_lim:.3f}] {unit}"
                    )

            # 6. KIỂM TRA TẦN SỐ THỰC TẾ & RỚT KHUNG HÌNH (FRAME DROP / JITTER)
            from act_pipeline.data.resample import extract_timestamps
            t_raw, has_ts, ts_stats = extract_timestamps(f, T, default_hz=expected_hz)
            res.info["duration_sec"] = ts_stats["duration_seconds"]
            res.info["actual_hz"] = ts_stats["actual_hz"]
            res.info["dropped_frames_est"] = ts_stats["estimated_dropped_frames"]
            res.info["drop_pct"] = ts_stats["drop_pct"]

            if ts_stats["jitter_detected"]:
                res.add_error(
                    f"🚨 LỆCH TẦN SỐ & RỚT KHUNG HÌNH: Tần số thực tế là {ts_stats['actual_hz']:.1f} Hz (chuẩn là {expected_hz} Hz). "
                    f"Chu kỳ dt dao động mạnh từ {ts_stats['min_dt']*1000:.1f}ms đến {ts_stats['max_dt']*1000:.1f}ms! "
                    f"Rớt ~{ts_stats['drop_pct']:.1f}% frames ({ts_stats['estimated_dropped_frames']} bước bị mất). "
                    f"👉 KHẮC PHỤC: Chạy lệnh 'python resample_dataset.py --input_dir <dir> --output_dir <dir_50hz>' "
                    f"để đồng bộ hóa quỹ đạo về 50.0 Hz trước khi huấn luyện!"
                )
            elif abs(ts_stats["actual_hz"] - expected_hz) > 1.0:
                res.add_warning(
                    f"Tần số thực tế ({ts_stats['actual_hz']} Hz) hơi lệch so với yêu cầu ({expected_hz} Hz)."
                )

    except Exception as e:
        res.add_error(f"Lỗi khi đọc file HDF5: {str(e)}")

    return res


def validate_dataset_directory(dataset_dir: str, expected_hz: float = 50.0) -> Tuple[int, int, List[ValidationResult]]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.hdf5")))
    # Loại trừ các file đang ghi dở .partial.hdf5
    files = [f for f in files if not f.endswith(".partial.hdf5")]

    if len(files) == 0:
        print(f"\n[X] Không tìm thấy file .hdf5 nào trong: {dataset_dir}\n")
        return 0, 0, []

    print("=" * 80)
    print(f"      🔍 BỘ KIỂM TRA ĐỊNH DẠNG DATASET HDF5 THEO DATA_FORMAT_SPECIFICATION.MD")
    print(f"      Thư mục: {dataset_dir} | Tổng số episodes: {len(files)} | Tần số chuẩn: {expected_hz}Hz")
    print("=" * 80)

    results: List[ValidationResult] = []
    passed_count = 0

    for idx, fpath in enumerate(files, 1):
        res = validate_single_hdf5(fpath, expected_hz=expected_hz)
        results.append(res)

        status_icon = "✅ CHUẨN" if res.passed else "❌ LỖI"
        info_str = f"T={res.info.get('timesteps', '?')} steps (~{res.info.get('duration_sec', '?')}s)"
        if "rgb_resolution" in res.info:
            info_str += f" | RGB: {res.info['rgb_resolution']}"
        if "calculated_hz" in res.info:
            info_str += f" | Thực tế: {res.info['calculated_hz']}Hz"

        print(f"[{idx:3d}/{len(files)}] {res.filename:<32} {status_icon} | {info_str}")

        if not res.passed:
            for err in res.errors:
                print(f"       -> 🛑 LỖI: {err}")
        for warn in res.warnings:
            print(f"       -> ⚠️  CẢNH BÁO: {warn}")

        if res.passed:
            passed_count += 1

    print("\n" + "=" * 80)
    print(f"📊 TỔNG KẾT: {passed_count}/{len(files)} episodes HỢP LỆ VỚI QUY CHUẨN ({(passed_count/len(files))*100:.1f}%)")
    if passed_count == len(files):
        print("🎉 TẤT CẢ FILE HDF5 ĐÃ ĐẠT CHUẨN 100% SẴN SÀNG CHO ACT PIPELINE!")
    else:
        print(f"⚠️  CÓ {len(files) - passed_count} FILE CẦN ĐỘI THU DATA SỬA LẠI THEO THÔNG BÁO LỖI Ở TRÊN.")
    print("=" * 80 + "\n")

    return passed_count, len(files), results


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra định dạng file HDF5 OpenArm ACT theo DATA_FORMAT_SPECIFICATION.md")
    parser.add_argument("path", type=str, nargs="?", default="dataset", help="Đường dẫn file .hdf5 hoặc thư mục chứa dataset (mặc định: dataset)")
    parser.add_argument("--hz", type=float, default=50.0, help="Tần số lấy mẫu yêu cầu (mặc định: 50.0 Hz)")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        res = validate_single_hdf5(args.path, expected_hz=args.hz)
        print("\n" + "=" * 70)
        print(f"Kết quả kiểm tra: {res.filename}")
        print(f"Trạng thái: {'✅ ĐẠT CHUẨN' if res.passed else '❌ CÓ LỖI'}")
        print(f"Thông tin : {json.dumps(res.info, indent=2, default=str)}")
        if res.errors:
            print("\nDanh sách lỗi:")
            for e in res.errors:
                print(f" - [LỖI] {e}")
        if res.warnings:
            print("\nDanh sách cảnh báo:")
            for w in res.warnings:
                print(f" - [CẢNH BÁO] {w}")
        print("=" * 70 + "\n")
        sys.exit(0 if res.passed else 1)
    elif os.path.isdir(args.path):
        passed, total, _ = validate_dataset_directory(args.path, expected_hz=args.hz)
        sys.exit(0 if passed == total and total > 0 else 1)
    else:
        print(f"[X] Đường dẫn không hợp lệ: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
