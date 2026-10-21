#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Episodic Dataset Loader for ACT Bimanual OpenArm
Supports HDF5 files containing 16-DOF joint data and RGB-D camera feeds.
Handles future action chunking [50, 16], boundary padding with is_pad mask,
and Z-score normalization for actions and qpos.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from act_pipeline.data.preprocess import preprocess_rgbd
from act_pipeline.data.normalization import normalize_data
from act_pipeline.data.resample import extract_timestamps, compute_resampling_plan, resample_signals


class BimanualEpisodicDataset(Dataset):
    """
    Episodic Dataset cho ACT:
    - Đọc các file HDF5 đại diện cho từng Episode.
    - Tự động phát hiện và khắc phục rớt khung hình (Frame Drop) và lệch tần số (Jitter)
      bằng cơ chế lấy mẫu lại đồng đều về chuẩn 50.0 Hz (control_dt = 0.02s).
    - Cắt lát dữ liệu thành các mẫu: (image_t, qpos_t, actions_{t:t+k}, is_pad).
    - Quản lý padding mask chính xác khi gần cuối Episode.
    - Chuẩn hóa Z-score toàn bộ vector góc khớp và hành động.
    """
    def __init__(
        self,
        file_paths: List[str],
        stats: Dict[str, np.ndarray],
        chunk_size: int = 50,
        step_stride: int = 1,
        target_img_size: Optional[Tuple[int, int]] = (480, 640),
        depth_min_m: float = 0.2,
        depth_max_m: float = 1.2,
        target_hz: float = 50.0,
        auto_resample: bool = True,
    ):
        super().__init__()
        self.file_paths = sorted(file_paths)
        if len(self.file_paths) == 0:
            raise ValueError("Không có file HDF5 nào được cung cấp cho Dataset!")

        self.stats = stats
        self.chunk_size = chunk_size
        self.step_stride = step_stride
        self.target_img_size = target_img_size
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.target_hz = target_hz
        self.auto_resample = auto_resample

        self.samples: List[Tuple[int, int]] = [] # (file_idx, start_t)
        self.episode_lengths: List[int] = []

        # Bộ nhớ đệm cho các episode cần lấy mẫu lại đồng đều (On-the-fly Resampling Cache)
        self.episode_resampled: Dict[int, bool] = {}
        self.episode_frame_indices: Dict[int, np.ndarray] = {}
        self.cached_qpos: Dict[int, np.ndarray] = {}
        self.cached_actions: Dict[int, np.ndarray] = {}

        self._build_index()

    def _build_index(self):
        """
        Quét nhanh độ dài từng episode để lập chỉ mục các lát cắt (file_idx, start_t).
        Tự động phát hiện hiện tượng rớt khung hình / giật lag thời gian và kích hoạt
        nội suy thời gian đồng đều 50 Hz.
        """
        total_steps = 0
        resampled_episodes = 0

        for file_idx, fpath in enumerate(self.file_paths):
            with h5py.File(fpath, "r") as f:
                T_raw = len(f["action"])

                # Kiểm tra timestamps và tần số thực tế
                t_raw, has_ts, ts_stats = extract_timestamps(f, T_raw, default_hz=self.target_hz)

                # Tự động phát hiện kích thước ảnh gốc nếu target_img_size là None
                if self.target_img_size is None and "observations/images" in f:
                    obs_img = f["observations/images"]
                    for k in ["chest_rgb", "chest", "front_rgb", "front"]:
                        if k in obs_img:
                            shape = obs_img[k].shape
                            self.target_img_size = (shape[1], shape[2])
                            break

                # Nếu cần lấy mẫu lại đồng đều về 50Hz do phát hiện rớt khung hình / lệch tần số
                need_resample = self.auto_resample and ts_stats["jitter_detected"]

                if need_resample:
                    t_uniform, frame_indices = compute_resampling_plan(t_raw, target_hz=self.target_hz)
                    T_ep = len(t_uniform)

                    # Nội suy và chuẩn hóa qpos, actions cho episode này
                    raw_qpos = f["observations/qpos"][:].astype(np.float32)
                    raw_actions = f["action"][:].astype(np.float32)

                    resampled_qpos = resample_signals(t_raw, raw_qpos, t_uniform)
                    resampled_actions = resample_signals(t_raw, raw_actions, t_uniform)

                    norm_qpos = normalize_data(resampled_qpos, self.stats["qpos_mean"], self.stats["qpos_std"])
                    norm_actions = normalize_data(resampled_actions, self.stats["action_mean"], self.stats["action_std"])

                    self.episode_resampled[file_idx] = True
                    self.episode_frame_indices[file_idx] = frame_indices
                    self.cached_qpos[file_idx] = norm_qpos
                    self.cached_actions[file_idx] = norm_actions
                    resampled_episodes += 1
                else:
                    self.episode_resampled[file_idx] = False
                    T_ep = T_raw

                self.episode_lengths.append(T_ep)
                total_steps += T_ep

                # Lấy tất cả các thời điểm start_t với bước nhảy step_stride
                for start_t in range(0, T_ep, self.step_stride):
                    self.samples.append((file_idx, start_t))

        resample_info = f", đồng bộ {resampled_episodes} episodes rớt frame về 50Hz" if resampled_episodes > 0 else ""
        print(f"[✓] Đã lập chỉ mục Dataset: {len(self.file_paths)} episodes, {total_steps} timesteps, {len(self.samples)} chunks (k={self.chunk_size}, stride={self.step_stride}{resample_info}).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_t = self.samples[idx]
        fpath = self.file_paths[file_idx]
        is_resampled = self.episode_resampled.get(file_idx, False)

        with h5py.File(fpath, "r") as f:
            T_ep = self.episode_lengths[file_idx]

            # Xác định frame camera tương ứng tại thời điểm start_t
            if is_resampled:
                cam_frame_idx = int(self.episode_frame_indices[file_idx][start_t])
            else:
                cam_frame_idx = start_t

            # 1. ĐỌC VÀ TIỀN XỬ LÝ ẢNH RGB-D TẠI THỜI ĐIỂM cam_frame_idx CHO CẢ 2 CAMERA
            obs_img = f["observations/images"]

            # --- Camera 1: CHEST (Trước ngực) ---
            chest_rgb_raw = None
            chest_depth_raw = None

            if "chest_rgb" in obs_img and "chest_depth" in obs_img:
                chest_rgb_raw = obs_img["chest_rgb"][cam_frame_idx]
                chest_depth_raw = obs_img["chest_depth"][cam_frame_idx]
            elif "chest" in obs_img:
                raw_chest = obs_img["chest"][cam_frame_idx]
                if raw_chest.shape[-1] == 4:
                    chest_rgb_raw = raw_chest[..., :3]
                    chest_depth_raw = raw_chest[..., 3]
                else:
                    chest_rgb_raw = raw_chest
                    if "chest_depth" in obs_img:
                        chest_depth_raw = obs_img["chest_depth"][cam_frame_idx]
            else:
                first_key = list(obs_img.keys())[0]
                chest_rgb_raw = obs_img[first_key][cam_frame_idx]

            chest_rgbd_tensor = preprocess_rgbd(
                rgb_raw=chest_rgb_raw,
                depth_raw=chest_depth_raw,
                depth_min_m=self.depth_min_m,
                depth_max_m=self.depth_max_m,
                target_size=self.target_img_size,
                preserve_aspect_ratio=True,
            )

            # --- Camera 2: FRONT (Đối diện nhìn robot thao tác) ---
            front_rgb_raw = None
            front_depth_raw = None

            if "front_rgb" in obs_img and "front_depth" in obs_img:
                front_rgb_raw = obs_img["front_rgb"][cam_frame_idx]
                front_depth_raw = obs_img["front_depth"][cam_frame_idx]
            elif "front" in obs_img:
                raw_front = obs_img["front"][cam_frame_idx]
                if raw_front.shape[-1] == 4:
                    front_rgb_raw = raw_front[..., :3]
                    front_depth_raw = raw_front[..., 3]
                else:
                    front_rgb_raw = raw_front
                    if "front_depth" in obs_img:
                        front_depth_raw = obs_img["front_depth"][cam_frame_idx]
            elif "front_rgb" in obs_img:
                front_rgb_raw = obs_img["front_rgb"][cam_frame_idx]

            if front_rgb_raw is not None:
                front_rgbd_tensor = preprocess_rgbd(
                    rgb_raw=front_rgb_raw,
                    depth_raw=front_depth_raw,
                    depth_min_m=self.depth_min_m,
                    depth_max_m=self.depth_max_m,
                    target_size=self.target_img_size,
                    preserve_aspect_ratio=True,
                )
            else:
                # Fallback dự phòng nếu dataset cũ chỉ có 1 camera chest: sao chép chest
                front_rgbd_tensor = chest_rgbd_tensor.clone()

            # Ghép 2 camera thành tensor 5D khi collate: [2, 4, H, W]
            image_tensor = torch.stack([chest_rgbd_tensor, front_rgbd_tensor], dim=0)

            # 2. ĐỌC VÀ CHUẨN HÓA GÓC KHỚP qpos TẠI THỜI ĐIỂM start_t
            if is_resampled:
                norm_qpos = self.cached_qpos[file_idx][start_t]
            else:
                raw_qpos = f["observations/qpos"][start_t].astype(np.float32)
                norm_qpos = normalize_data(raw_qpos, self.stats["qpos_mean"], self.stats["qpos_std"])

            qpos_tensor = torch.tensor(norm_qpos, dtype=torch.float32)

            # 3. ĐỌC VÀ CHUẨN HÓA CHUỖI HÀNH ĐỘNG ACTION CHUNK [start_t : start_t + chunk_size]
            end_t = start_t + self.chunk_size
            is_pad = np.zeros(self.chunk_size, dtype=bool)

            if is_resampled:
                actions_data = self.cached_actions[file_idx]
                if end_t <= T_ep:
                    norm_actions = actions_data[start_t:end_t]
                else:
                    available = actions_data[start_t:T_ep]
                    num_available = len(available)
                    pad_count = self.chunk_size - num_available
                    last_act = available[-1:]
                    padding = np.repeat(last_act, pad_count, axis=0)
                    norm_actions = np.concatenate([available, padding], axis=0)
                    is_pad[num_available:] = True
            else:
                if end_t <= T_ep:
                    raw_actions = f["action"][start_t:end_t].astype(np.float32)
                else:
                    available = f["action"][start_t:T_ep].astype(np.float32)
                    num_available = len(available)
                    pad_count = self.chunk_size - num_available
                    last_act = available[-1:]
                    padding = np.repeat(last_act, pad_count, axis=0)
                    raw_actions = np.concatenate([available, padding], axis=0)
                    is_pad[num_available:] = True
                norm_actions = normalize_data(raw_actions, self.stats["action_mean"], self.stats["action_std"])

            actions_tensor = torch.tensor(norm_actions, dtype=torch.float32)
            is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)

        return {
            "image": image_tensor,              # [2, 4, H, W] (02 Camera: Chest + Front)
            "image_chest": chest_rgbd_tensor,   # [4, H, W]
            "image_front": front_rgbd_tensor,   # [4, H, W]
            "qpos": qpos_tensor,                # [16]
            "actions": actions_tensor,          # [chunk_size, 16]
            "is_pad": is_pad_tensor,            # [chunk_size]
        }
