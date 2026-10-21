#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration parameters for ACT (Action Chunking with Transformers)
Bimanual OpenArm (16-DOF: 2 arms x 8 motors) + 02 RGB-D Cameras (Chest + Front, 480x640)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ModelConfig:
    # Vision Backbone (Camera 480x640: Chest + Front)
    in_channels: int = 4            # RGB (3) + Depth (1) cho mỗi camera
    num_cameras: int = 2            # 02 Camera: Chest (trước ngực) + Front (đối diện)
    camera_names: Tuple[str, ...] = ("chest", "front")
    backbone_type: str = "yolov8_nano" # "yolov8_nano" (khuyên dùng cho nhặt đồ/detection), "yolov8_small", "resnet18", "resnet34", "resnet50", "convnext_nano"
    freeze_backbone: bool = True    # Đóng băng pretrained backbone (chỉ train stem/conv1 adapter & projection)
    yolo_stride: int = 16           # Stride cho Backbone (16 = 1200 tokens/cam; 32 = 300 tokens/cam @ 480x640)
    img_height: int = 480           # Chiều cao ảnh chuẩn hóa (H=480, W=640)
    img_width: int = 640
    
    # Dimensions
    d_model: int = 512              # Transformer hidden dimension
    nheads: int = 8                 # Multi-head attention heads
    dim_feedforward: int = 2048     # FFN intermediate dimension
    dropout: float = 0.1
    
    # CVAE
    cvae_layers: int = 2            # Số layer TransformerEncoder của CVAE
    latent_dim: int = 32            # Kích thước latent vector z
    
    # Policy Decoder (Chuẩn Stanford ALOHA ACT sử dụng 7 layers)
    decoder_layers: int = 7         # Số layer TransformerDecoder của Policy (Stanford ALOHA standard = 7)
    chunk_size: int = 50            # Action chunk size k=50 (1 giây tại 50Hz)
    action_dim: int = 16            # 16 khớp (8 tay trái: J1..J7 + gripper; 8 tay phải: J1..J7 + gripper)
    qpos_dim: int = 16              # 16 góc khớp hiện tại
    
    # Depth Preprocessing
    depth_min_m: float = 0.2        # Khoảng cách bàn tối thiểu (mét)
    depth_max_m: float = 1.2        # Khoảng cách bàn tối đa (mét)

    # 3D Point & Ray Features (Plücker Ray Tokens)
    use_3d_features: bool = False   # Cờ bật/tắt 3D Point Tokens (mặc định False: chỉ chạy backbone đã chọn)
    point_grid_h: int = 15          # Chiều cao lưới điểm 3D (15x20 = 300 points/cam)
    point_grid_w: int = 20          # Chiều rộng lưới điểm 3D
    include_rgb_in_3d: bool = True  # Kèm màu RGB vào vector đặc trưng điểm 3D


@dataclass
class TrainConfig:
    # Paths
    dataset_dir: str = "data_set/pick_and_place"
    output_dir: str = "checkpoints/act_openarm"
    stats_file: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    
    # Training Loop
    epochs: int = 500
    batch_size: int = 16
    lr: float = 1e-4
    lr_backbone: float = 1e-5       # Learning rate nhỏ cho conv1 adapter
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"    # "cosine", "step", "none"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Loss Weights
    kl_weight: float = 10.0         # Trọng số beta cho KL Divergence loss
    loss_type: str = "l1"           # "l1" hoặc "l2"
    
    # Hardware & Performance
    num_workers: int = 2
    use_amp: bool = True            # Automatic Mixed Precision (FP16)
    seed: int = 42
    device: str = "cuda"            # "cuda" hoặc "cpu"
    
    # Validation & Logging
    val_split: float = 0.15         # 15% validation data
    eval_every: int = 10            # Đánh giá validation mỗi N epochs (mặc định 10)
    save_every: int = 10            # Lưu checkpoint định kỳ mỗi N epochs (mặc định 10)
    log_every_steps: int = 10       # In progress mỗi N batches
    target_hz: float = 50.0         # Tần số điều khiển chuẩn (50Hz = dt=0.02s)
    auto_resample: bool = True      # Tự động phát hiện rớt frame và đồng bộ về 50Hz
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "act-openarm"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None # API key xác thực WandB (tùy chọn)
    wandb_name: Optional[str] = None    # Tên Run hiển thị trên WandB
    
    # TensorBoard nâng cao
    tb_histogram_every: int = 10    # Ghi histogram mỗi N epochs
    tb_image_every: int = 10        # Ghi ảnh mẫu mỗi N epochs
    tb_embedding_every: int = 50    # Ghi CVAE embedding mỗi N epochs



@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/act_openarm/best_checkpoint.onnx"
    dataset_dir: str = "data_set/pick_and_place"
    output_dir: str = "evaluation_results"
    batch_size: int = 16
    num_workers: int = 2
    temporal_ensemble: bool = True
    ensemble_m: float = 0.01        # Hệ số suy giảm trọng số exp(-m * i)
    device: str = "cuda"
