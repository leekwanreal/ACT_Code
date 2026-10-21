#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalization Statistics Module
Calculates, saves, loads and applies Z-score normalization for qpos and actions.
Formula: x_norm = (x - mean) / std
Unnormalize: x = x_norm * std + mean
"""

import os
import glob
import pickle
import json
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
try:
    import h5py
except ImportError:
    h5py = None


def compute_norm_stats(file_paths: List[str]) -> Dict[str, np.ndarray]:
    """
    Quét toàn bộ danh sách file HDF5 để tính mean và std của qpos và action.
    """
    if h5py is None:
        raise ImportError("Cần cài đặt h5py để đọc file HDF5: pip install h5py")

    if len(file_paths) == 0:
        raise ValueError("Danh sách file rỗng, không thể tính thống kê chuẩn hóa!")

    all_qpos = []
    all_actions = []

    print(f"[*] Đang quét {len(file_paths)} file HDF5 để tính thống kê chuẩn hóa (Mean & Std)...")
    for fpath in file_paths:
        with h5py.File(fpath, "r") as f:
            all_qpos.append(f["observations/qpos"][:])
            all_actions.append(f["action"][:])

    all_qpos = np.concatenate(all_qpos, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Đảm bảo std không bị chia cho 0 với epsilon 1e-4
    qpos_std = np.std(all_qpos, axis=0)
    qpos_std = np.clip(qpos_std, 1e-4, np.inf).astype(np.float32)

    action_std = np.std(all_actions, axis=0)
    action_std = np.clip(action_std, 1e-4, np.inf).astype(np.float32)

    stats = {
        "qpos_mean": np.mean(all_qpos, axis=0).astype(np.float32),
        "qpos_std": qpos_std,
        "action_mean": np.mean(all_actions, axis=0).astype(np.float32),
        "action_std": action_std,
        "num_episodes": len(file_paths),
        "total_timesteps": len(all_qpos),
    }

    print(f"[✓] Đã tính xong thống kê trên tổng số {len(all_qpos)} timesteps.")
    print(f"    - qpos_mean shape: {stats['qpos_mean'].shape}, range: [{stats['qpos_mean'].min():.3f}, {stats['qpos_mean'].max():.3f}]")
    print(f"    - action_mean shape: {stats['action_mean'].shape}, range: [{stats['action_mean'].min():.3f}, {stats['action_mean'].max():.3f}]")
    return stats


def save_norm_stats(stats: Dict[str, Any], save_path: str):
    """
    Lưu thống kê chuẩn hóa thành file .pkl hoặc .pt
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if save_path.endswith(".pkl"):
        with open(save_path, "wb") as f:
            pickle.dump(stats, f)
    elif save_path.endswith(".pt") or save_path.endswith(".pth"):
        torch.save(stats, save_path)
    elif save_path.endswith(".json"):
        json_dict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in stats.items()
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)
    else:
        with open(save_path, "wb") as f:
            pickle.dump(stats, f)
    print(f"[✓] Đã lưu thống kê chuẩn hóa tại: {save_path}")


def load_norm_stats(load_path: str) -> Dict[str, np.ndarray]:
    """
    Đọc thống kê chuẩn hóa từ file .pkl, .pt, .pth hoặc .json
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Không tìm thấy file stats tại: {load_path}")

    if load_path.endswith(".pt") or load_path.endswith(".pth"):
        stats = torch.load(load_path, map_location="cpu", weights_only=False)
        if "stats" in stats and isinstance(stats["stats"], dict):
            stats = stats["stats"]
    elif load_path.endswith(".json"):
        with open(load_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        stats = {
            k: np.array(v, dtype=np.float32) if isinstance(v, list) else v
            for k, v in raw.items()
        }
    else:
        with open(load_path, "rb") as f:
            stats = pickle.load(f)

    # Chuyển đổi sang np.ndarray nếu lưu dưới dạng Tensor
    for key in ["qpos_mean", "qpos_std", "action_mean", "action_std"]:
        if key in stats and isinstance(stats[key], torch.Tensor):
            stats[key] = stats[key].cpu().numpy().astype(np.float32)

    return stats


def normalize_data(
    data: Union[np.ndarray, torch.Tensor],
    mean: Union[np.ndarray, torch.Tensor],
    std: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Chuẩn hóa Z-score: (x - mean) / std
    """
    return (data - mean) / std


def unnormalize_data(
    data: Union[np.ndarray, torch.Tensor],
    mean: Union[np.ndarray, torch.Tensor],
    std: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Giải chuẩn hóa Z-score: x_norm * std + mean
    """
    return data * std + mean
