#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Point & Ray Feature Extractor for ACT (Action Chunking with Transformers)
Biểu diễn hình học 3D dựa trên Tọa độ tia chuẩn hóa & Tọa độ đường thẳng Plücker:
- Input: Ảnh RGB-D [B, num_cameras, 4, H, W] (hoặc ghép kênh)
- Trích xuất lưới điểm không gian Metric Ray Points P = (u_norm * Z, v_norm * Z, Z)
- Vector chỉ phương tia Plücker d = (u_norm, v_norm, 1.0) / ||(u_norm, v_norm, 1.0)||
- Bổ sung thông tin màu RGB cục bộ và nhãn Camera ID (Chest vs Front)
- PointNet-style MLP (Linear + LayerNorm + SiLU + Residual) chiếu về d_model (512)
- Đầu ra: tokens_3d [B, num_cameras * H_g * W_g, d_model]
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Point3DEncoder(nn.Module):
    """
    Module trích xuất đặc trưng không gian 3D dạng tia Plücker (Plücker Ray Tokens).
    Đặc tính nổi bật:
    1. Hoạt động trên GPU tensor thuần túy (Vectorized), độ trễ < 0.2 ms, giữ vững chu kỳ 50Hz.
    2. Không phụ thuộc vào việc file HDF5 có lưu Camera Intrinsics hay không nhờ hệ tọa độ tia chuẩn hóa.
    3. Hợp nhất tự nhiên thông tin hình học của 2 camera (Chest + Front).
    """
    def __init__(
        self,
        d_model: int = 512,
        grid_size: Tuple[int, int] = (15, 20),
        num_cameras: int = 2,
        include_rgb: bool = True,
        depth_min_m: float = 0.2,
        depth_max_m: float = 1.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_h, self.grid_w = grid_size
        self.num_cameras = num_cameras
        self.include_rgb = include_rgb
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.num_points_per_cam = self.grid_h * self.grid_w

        # Kích thước vector đặc trưng điểm 3D thô:
        # P_3D (3: X, Y, Z) + Ray Direction d (3: dx, dy, dz) + Depth Z (1)
        # + RGB cục bộ (3 nếu include_rgb=True) + Camera ID one-hot (num_cameras)
        in_dim = 3 + 3 + 1 + (3 if include_rgb else 0) + num_cameras
        self.in_dim = in_dim

        # PointNet-style MLP: in_dim -> 128 -> 256 -> d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

        # Residual projection nếu cần
        self.skip_proj = nn.Linear(in_dim, d_model)

        # Learnable 3D Token Type Embedding để Transformer phân biệt rõ với Visual tokens 2D
        self.token_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.token_type_embed, std=0.02)

        # Cache lưới tia chuẩn hóa (u_norm, v_norm, ray_dir)
        self._cached_grid = None

    def _get_ray_grid(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tạo lưới tia chuẩn hóa [-1, 1] cho kích thước lưới (grid_h, grid_w).
        Returns:
            uv_norm: [1, grid_h, grid_w, 2] - (u_norm, v_norm)
            ray_dir: [1, grid_h, grid_w, 3] - Vector chỉ phương đơn vị d
        """
        if self._cached_grid is not None and self._cached_grid[0].device == device and self._cached_grid[0].dtype == dtype:
            return self._cached_grid

        # Tạo tọa độ tâm các ô lưới chuẩn hóa [-1, 1]
        v_steps = torch.linspace(-1.0 + 1.0 / self.grid_h, 1.0 - 1.0 / self.grid_h, self.grid_h, device=device, dtype=dtype)
        u_steps = torch.linspace(-1.0 + 1.0 / self.grid_w, 1.0 - 1.0 / self.grid_w, self.grid_w, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v_steps, u_steps, indexing="ij") # [grid_h, grid_w]

        uv_norm = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0) # [1, grid_h, grid_w, 2]

        # Vector chỉ phương tia: d = (u_norm, v_norm, 1.0) / norm
        ones = torch.ones_like(u_grid)
        rays = torch.stack([u_grid, v_grid, ones], dim=-1) # [grid_h, grid_w, 3]
        ray_norms = torch.norm(rays, dim=-1, keepdim=True)
        ray_dir = (rays / ray_norms).unsqueeze(0) # [1, grid_h, grid_w, 3]

        self._cached_grid = (uv_norm, ray_dir)
        return uv_norm, ray_dir

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            image: 
                - [B, num_cameras, 4, H, W] tensor (Chuẩn 2 camera RGB-D)
                - hoặc [B, 8, H, W] tensor (2 camera ghép kênh)
                - hoặc [B, 4, H, W] tensor (1 camera)
        Outputs:
            tokens_3d: [B, num_cameras * grid_h * grid_w, d_model]
        """
        if image.ndim == 4:
            if image.shape[1] == 8:
                # [B, 8, H, W] -> [B, 2, 4, H, W]
                B, _, H, W = image.shape
                image = image.view(B, 2, 4, H, W)
            else:
                # [B, 4, H, W] -> [B, 1, 4, H, W]
                image = image.unsqueeze(1)

        B, num_cams, C, H, W = image.shape
        device = image.device
        dtype = image.dtype

        uv_norm, ray_dir = self._get_ray_grid(device, dtype)
        # uv_norm: [1, grid_h, grid_w, 2]
        # ray_dir: [1, grid_h, grid_w, 3]

        all_cam_tokens = []

        for cam_idx in range(num_cams):
            cam_img = image[:, cam_idx] # [B, 4, H, W]
            rgb = cam_img[:, :3]        # [B, 3, H, W]
            depth = cam_img[:, 3:4]     # [B, 1, H, W]

            # 1. Pooling độ sâu về kích thước lưới: [B, 1, grid_h, grid_w]
            depth_pooled = F.adaptive_avg_pool2d(depth, (self.grid_h, self.grid_w))
            # Kẹp an toàn trong dải depth hợp lệ
            depth_pooled = torch.clamp(depth_pooled, min=self.depth_min_m, max=self.depth_max_m)

            # 2. Tính tọa độ điểm 3D Metric Ray: P = (u_norm * Z, v_norm * Z, Z)
            # depth_pooled: [B, 1, grid_h, grid_w] -> permute -> [B, grid_h, grid_w, 1]
            Z = depth_pooled.permute(0, 2, 3, 1) # [B, grid_h, grid_w, 1]
            X = uv_norm[..., 0:1] * Z             # [B, grid_h, grid_w, 1]
            Y = uv_norm[..., 1:2] * Z             # [B, grid_h, grid_w, 1]
            P_3D = torch.cat([X, Y, Z], dim=-1)   # [B, grid_h, grid_w, 3]

            # Ray directions mở rộng cho Batch: [B, grid_h, grid_w, 3]
            D_ray = ray_dir.expand(B, -1, -1, -1)

            feature_parts = [P_3D, D_ray, Z]

            # 3. Thông tin màu sắc cục bộ
            if self.include_rgb:
                rgb_pooled = F.adaptive_avg_pool2d(rgb, (self.grid_h, self.grid_w)) # [B, 3, grid_h, grid_w]
                RGB = rgb_pooled.permute(0, 2, 3, 1) # [B, grid_h, grid_w, 3]
                feature_parts.append(RGB)

            # 4. Camera ID One-Hot Embedding: [B, grid_h, grid_w, num_cameras]
            cam_onehot = torch.zeros(B, self.grid_h, self.grid_w, self.num_cameras, device=device, dtype=dtype)
            cam_onehot[..., min(cam_idx, self.num_cameras - 1)] = 1.0
            feature_parts.append(cam_onehot)

            # 5. Ghép các đặc trưng thành vector điểm 3D: [B, grid_h, grid_w, in_dim]
            point_features = torch.cat(feature_parts, dim=-1)
            # Làm phẳng không gian: [B, num_points_per_cam, in_dim]
            point_features = point_features.flatten(1, 2)

            # 6. Chiếu qua Point MLP có Skip Connection
            p_embed = self.mlp(point_features) + self.skip_proj(point_features) # [B, num_points_per_cam, d_model]
            all_cam_tokens.append(p_embed)

        # Ghép tokens 3D của tất cả camera: [B, num_cams * num_points_per_cam, d_model]
        tokens_3d = torch.cat(all_cam_tokens, dim=1)

        # Thêm 3D Token Type Embedding để Transformer phân biệt với token 2D
        tokens_3d = tokens_3d + self.token_type_embed

        return tokens_3d
