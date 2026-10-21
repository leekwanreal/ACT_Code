#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to create matplotlib figures for TensorBoard visualization.
Strict error handling (Zero-Bypass): mọi tham số không hợp lệ đều chủ động raise ValueError.
"""

from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Backend không phụ thuộc màn hình hiển thị (headless)
import matplotlib.pyplot as plt
import numpy as np

# 16 khớp robot OpenArm: 8 khớp tay trái và 8 khớp tay phải
JOINT_NAMES = [
    "L_J1_Base", "L_J2_Shoulder", "L_J3_Elbow_P", "L_J4_Elbow_R",
    "L_J5_Wrist_P", "L_J6_Wrist_R", "L_J7_Wrist_Y", "L_J8_Gripper",
    "R_J1_Base", "R_J2_Shoulder", "R_J3_Elbow_P", "R_J4_Elbow_R",
    "R_J5_Wrist_P", "R_J6_Wrist_R", "R_J7_Wrist_Y", "R_J8_Gripper",
]


def plot_per_joint_mae(per_joint_mae: Union[List[float], np.ndarray], epoch: int = 0) -> matplotlib.figure.Figure:
    """
    Vẽ biểu đồ cột sai số MAE (chuyển đổi sang Degree) cho 16 khớp robot OpenArm (16-DOF).
    
    Args:
        per_joint_mae: Danh sách hoặc mảng 16 phần tử chứa sai số trung bình (radian) cho từng khớp.
        epoch: Epoch hiện tại để gắn nhãn tiêu đề biểu đồ.
        
    Returns:
        matplotlib.figure.Figure: Đối tượng figure của biểu đồ đã vẽ.
        
    Raises:
        ValueError: Nếu per_joint_mae là None, không có độ dài, hoặc độ dài khác 16 phần tử.
    """
    if per_joint_mae is None:
        raise ValueError("plot_per_joint_mae(): per_joint_mae=None.")
    if not hasattr(per_joint_mae, "__len__"):
        raise ValueError("plot_per_joint_mae(): per_joint_mae phải có độ dài 16 phần tử.")
    if len(per_joint_mae) != 16:
        raise ValueError(
            f"plot_per_joint_mae(): Kỳ vọng 16 giá trị, nhận được {len(per_joint_mae)}."
        )

    mae_deg = []
    for v in per_joint_mae:
        if hasattr(v, "detach"):
            v = v.detach().cpu().item()
        elif hasattr(v, "item"):
            v = v.item()
        mae_deg.append(float(np.degrees(float(v))))

    fig, ax = plt.subplots(figsize=(14, 5))

    # 8 khớp tay trái màu xanh (#2196F3), 8 khớp tay phải màu cam (#FF9800)
    colors = ['#2196F3'] * 8 + ['#FF9800'] * 8
    ax.bar(range(16), mae_deg, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(16))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("MAE (Degrees)")
    ax.set_title(f"Per-Joint MAE — Epoch {epoch}")
    ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.5, label='Tuyệt vời (< 3°)')
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.5, label='Cần cải thiện (> 6°)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    raw_pred: np.ndarray,
    ensemble_pred: np.ndarray,
    joint_idx: int = 0,
) -> matplotlib.figure.Figure:
    """
    Vẽ biểu đồ so sánh quỹ đạo góc khớp theo thời gian (Ground Truth vs Raw Chunk vs Temporal Ensemble).
    
    Args:
        gt_actions: Mảng 2D quỹ đạo Ground Truth [T, action_dim] (radian).
        raw_pred: Mảng 2D dự đoán hành động thô [T, action_dim] (radian).
        ensemble_pred: Mảng 2D dự đoán đã làm mượt Temporal Ensemble [T, action_dim] (radian).
        joint_idx: Chỉ số khớp cần vẽ so sánh (0 <= joint_idx < action_dim).
        
    Returns:
        matplotlib.figure.Figure: Đối tượng figure của biểu đồ đã vẽ.
        
    Raises:
        ValueError: Nếu bất kỳ tham số nào là None, kích thước/shape không khớp,
                    mảng không có 2 chiều, mảng rỗng, hoặc joint_idx ngoài phạm vi.
    """
    if gt_actions is None:
        raise ValueError("plot_trajectory_comparison(): gt_actions=None.")
    if raw_pred is None:
        raise ValueError("plot_trajectory_comparison(): raw_pred=None.")
    if ensemble_pred is None:
        raise ValueError("plot_trajectory_comparison(): ensemble_pred=None.")

    # Chuyển đổi PyTorch Tensor sang NumPy ndarray nếu có
    if hasattr(gt_actions, "detach"):
        gt_actions = gt_actions.detach().cpu().numpy()
    elif not hasattr(gt_actions, "shape"):
        gt_actions = np.asarray(gt_actions)

    if hasattr(raw_pred, "detach"):
        raw_pred = raw_pred.detach().cpu().numpy()
    elif not hasattr(raw_pred, "shape"):
        raw_pred = np.asarray(raw_pred)

    if hasattr(ensemble_pred, "detach"):
        ensemble_pred = ensemble_pred.detach().cpu().numpy()
    elif not hasattr(ensemble_pred, "shape"):
        ensemble_pred = np.asarray(ensemble_pred)

    if gt_actions.shape != raw_pred.shape:
        raise ValueError(
            f"plot_trajectory_comparison(): shape không khớp: "
            f"gt={gt_actions.shape}, raw={raw_pred.shape}."
        )
    if gt_actions.shape != ensemble_pred.shape:
        raise ValueError(
            f"plot_trajectory_comparison(): shape không khớp: "
            f"gt={gt_actions.shape}, ensemble={ensemble_pred.shape}."
        )
    if len(gt_actions.shape) != 2:
        raise ValueError(
            f"plot_trajectory_comparison(): mảng phải có 2 chiều [T, num_joints], nhận được shape={gt_actions.shape}."
        )
    if gt_actions.shape[0] == 0:
        raise ValueError("plot_trajectory_comparison(): mảng rỗng (T=0).")
    if joint_idx < 0 or joint_idx >= gt_actions.shape[1]:
        raise ValueError(
            f"plot_trajectory_comparison(): joint_idx={joint_idx} ngoài phạm vi "
            f"[0, {gt_actions.shape[1]})."
        )

    fig, ax = plt.subplots(figsize=(12, 4))
    T = len(gt_actions)
    t = np.arange(T)

    ax.plot(t, np.degrees(gt_actions[:, joint_idx]), 'k-', label='Ground Truth', linewidth=1.5)
    ax.plot(t, np.degrees(raw_pred[:, joint_idx]), 'r-', alpha=0.6, label='Raw Chunk', linewidth=0.8)
    ax.plot(t, np.degrees(ensemble_pred[:, joint_idx]), 'b-', label='Temporal Ensemble', linewidth=1.0)

    ax.set_xlabel("Timestep (50Hz)")
    ax.set_ylabel("Angle (Degrees)")
    joint_name = JOINT_NAMES[joint_idx] if 0 <= joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
    ax.set_title(f"Trajectory Comparison — {joint_name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


__all__ = [
    "JOINT_NAMES",
    "plot_per_joint_mae",
    "plot_trajectory_comparison",
]
