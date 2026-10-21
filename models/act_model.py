#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ACT Policy Architecture for Bimanual OpenArm (16-DOF) + 02 RGB-D Cameras (Chest + Front)
Vision Backbone: YOLOv8 / ConvNeXt-Nano
- 02 Cameras RGB-D (Chest + Front) [B, 2, 4, 480, 640]
- YOLOv8 / ConvNeXt-Nano Backbone (Frozen hoặc Fine-tuned)
- 2D Sinusoidal PE (Fixed) -> Visual Tokens [B, num_cams * N_tokens, 512]
- Joint State Projection Linear 16 -> 512
- CVAE Transformer Encoder (2 Layers) + Latent Heads 512 -> 32
- Latent Projection Linear 32 -> 512
- Memory Tokens [B, N_memory, 512]
- 50 Action Queries + 1D Temporal Sinusoidal PE
- Transformer Decoder (7 Layers Stanford ALOHA standard)
- Action Linear Head 512 -> 16
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from act_pipeline.config import ModelConfig
from act_pipeline.models.backbone import build_vision_backbone, RGBDConvNeXtBackbone
from act_pipeline.models.cvae import CVAEEncoder
from act_pipeline.models.policy import TransformerPolicyDecoder
from act_pipeline.models.point_3d import Point3DEncoder


class ACTPolicy(nn.Module):
    """
    Toàn bộ mô hình ACT cho robot OpenArm hai tay (16-DOF):
    Ghép nối Backbone thị giác RGB-D (YOLOv8 / ConvNeXt-Nano), CVAE Latent Generator và Policy Decoder.
    Hỗ trợ đa camera: 02 Camera RGB-D (Chest + Front) chuẩn độ phân giải 480x640.
    Hỗ trợ cờ tùy chọn 3D Point Features (Plücker Ray Tokens).
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.d_model = config.d_model
        self.chunk_size = config.chunk_size
        self.action_dim = config.action_dim
        self.qpos_dim = config.qpos_dim
        self.latent_dim = config.latent_dim
        self.num_cameras = getattr(config, "num_cameras", 2)
        self.camera_names = getattr(config, "camera_names", ("chest", "front"))

        # 1. THỊ GIÁC (Vision Backbone: YOLOv8 / ConvNeXt-Nano)
        self.backbone = build_vision_backbone(
            backbone_type=getattr(config, "backbone_type", "yolov8_nano"),
            d_model=config.d_model,
            freeze_backbone=config.freeze_backbone,
            in_channels=config.in_channels,
            stride=getattr(config, "yolo_stride", 16),
        )

        # 1.1. CỜ BẬT/TẮT 3D POINT FEATURES (Plücker Ray Tokens)
        self.use_3d_features = getattr(config, "use_3d_features", False)
        if self.use_3d_features:
            grid_size = (getattr(config, "point_grid_h", 15), getattr(config, "point_grid_w", 20))
            self.point_3d_encoder = Point3DEncoder(
                d_model=config.d_model,
                grid_size=grid_size,
                num_cameras=self.num_cameras,
                include_rgb=getattr(config, "include_rgb_in_3d", True),
                depth_min_m=getattr(config, "depth_min_m", 0.2),
                depth_max_m=getattr(config, "depth_max_m", 1.2),
            )
        else:
            self.point_3d_encoder = None

        # 2. TRẠNG THÁI KHỚP (Joint State Projection)
        self.qpos_proj = nn.Linear(config.qpos_dim, config.d_model)

        # 3. CVAE (Latent Space)
        self.cvae = CVAEEncoder(
            d_model=config.d_model,
            nheads=config.nheads,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.cvae_layers,
            dropout=config.dropout,
            action_dim=config.action_dim,
            qpos_dim=config.qpos_dim,
            latent_dim=config.latent_dim,
        )

        # 4. POLICY DECODER
        self.policy_decoder = TransformerPolicyDecoder(
            d_model=config.d_model,
            nheads=config.nheads,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
            chunk_size=config.chunk_size,
            action_dim=config.action_dim,
        )

    def forward(
        self,
        image: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Inputs:
            image: 
                - [B, num_cameras, 4, H, W] tensor (Chuẩn 2 camera RGB-D: Chest và Front 480x640)
                - hoặc [B, 8, H, W] tensor (Ghép kênh 4 kênh chest + 4 kênh front)
                - hoặc [B, 4, H, W] tensor (Trường hợp 1 camera)
                - hoặc dict/list các camera tensor
            qpos: [B, 16] - Vector góc khớp 16 motor hiện tại
            actions: [B, chunk_size, 16] - Chuỗi hành động tương lai (Chỉ truyền khi Train)
            is_pad: [B, chunk_size] bool - Mask padding nếu gần cuối episode
        Outputs:
            pred_actions: [B, chunk_size, 16] - Quỹ đạo 50 bước dự đoán cho 16 motor
            mu: [B, latent_dim] hoặc None khi Inference
            logvar: [B, latent_dim] hoặc None khi Inference
        """
        if isinstance(image, dict):
            cam_list = [image[k] for k in self.camera_names if k in image]
            if len(cam_list) == 0:
                cam_list = list(image.values())
            image = torch.stack(cam_list, dim=1)
        elif isinstance(image, (list, tuple)):
            image = torch.stack(image, dim=1)

        B = qpos.shape[0]

        # 1. Trích xuất đặc trưng thị giác từ các camera
        if image.ndim == 5:
            # [B, num_cams, C, H, W]
            num_cams = image.shape[1]
            all_cam_tokens = []
            for cam_idx in range(num_cams):
                cam_feat = self.backbone(image[:, cam_idx])  # [B, N_tokens, d_model]
                all_cam_tokens.append(cam_feat)
            visual_tokens = torch.cat(all_cam_tokens, dim=1)  # [B, num_cams * N_tokens, d_model]
        elif image.ndim == 4:
            if image.shape[1] == 8:  # 2 cameras * 4 channels
                chest_tokens = self.backbone(image[:, :4])
                front_tokens = self.backbone(image[:, 4:])
                visual_tokens = torch.cat([chest_tokens, front_tokens], dim=1)
            else:
                visual_tokens = self.backbone(image)
        else:
            raise ValueError(f"image tensor phải có 4 hoặc 5 chiều, nhận được ndim={image.ndim}")

        # 2. Chiếu góc khớp hiện tại -> 1 qpos Token [B, 1, d_model]
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)

        # 3. CVAE: Lấy mẫu latent vector z
        # Train: z = mu + eps * sigma từ (actions, qpos)
        # Eval / Inference: z = 0
        z_token, mu, logvar = self.cvae(qpos=qpos, actions=actions, is_pad=is_pad)

        # 4. Trộn Memory Tokens:
        # Nếu có cờ use_3d_features: thêm tokens_3d vào chuỗi Memory
        # Nếu không có cờ: chỉ chạy visual_tokens từ backbone đã chọn
        memory_tokens = [visual_tokens]
        if self.point_3d_encoder is not None:
            tokens_3d = self.point_3d_encoder(image)
            memory_tokens.append(tokens_3d)
        memory_tokens.extend([qpos_token, z_token])
        memory = torch.cat(memory_tokens, dim=1)

        # 5. Giải mã Action Chunk qua Policy Decoder
        pred_actions = self.policy_decoder(memory)  # [B, chunk_size, action_dim]

        return pred_actions, mu, logvar

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        actions: torch.Tensor,
        is_pad: Optional[torch.Tensor],
        mu: Optional[torch.Tensor],
        logvar: Optional[torch.Tensor],
        kl_weight: float = 10.0,
        loss_type: str = "l1",
    ) -> Dict[str, torch.Tensor]:
        """
        Tính toán hàm mất mát Loss:
        Loss = L_recon (L1 hoặc L2 có padding mask) + beta * KL_Loss
        """
        B = actions.shape[0]

        # 1. Reconstruction Loss (L1 hoặc MSE có mask loại bỏ padding)
        if loss_type == "l1":
            raw_recon_loss = F.l1_loss(pred_actions, actions, reduction="none") # [B, chunk_size, action_dim]
        else:
            raw_recon_loss = F.mse_loss(pred_actions, actions, reduction="none")

        if is_pad is not None:
            # is_pad có shape [B, chunk_size], True là padding (bị loại)
            valid_mask = (~is_pad).unsqueeze(-1).float() # [B, chunk_size, 1]
            recon_loss = (raw_recon_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0) / self.action_dim
        else:
            recon_loss = raw_recon_loss.mean()

        # 2. KL Divergence Loss giữa q(z | actions, qpos) và p(z) ~ N(0, I)
        if mu is not None and logvar is not None:
            # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        else:
            kl_loss = torch.tensor(0.0, device=pred_actions.device)

        # 3. Tổng hàm mất mát
        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        image: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dự đoán Action Chunk trong quá trình Inference / Robot Deployment.
        Tự động gán cứng z = 0 và không tính gradient.
        Inputs:
            image: [B, 2, 4, H, W] (hoặc [B, 4, H, W] / [B, 8, H, W] / dict)
            qpos: [B, 16]
        Output:
            pred_actions: [B, chunk_size, 16]
        """
        self.eval()
        pred_actions, _, _ = self.forward(image=image, qpos=qpos, actions=None, is_pad=None)
        return pred_actions

    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Thống kê chi tiết số tham số Trainable vs Frozen theo đúng bảng ACT_ARCHITECTURE.md
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Thống kê tham số Backbone ConvNeXt-Nano
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        backbone_frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)

        summary = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_pct": (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0,
            "modules": {
                "backbone_type": getattr(self.config, "backbone_type", "convnext_nano"),
                "backbone_trainable": backbone_trainable,
                "backbone_frozen": backbone_frozen,
                "img_proj": sum(p.numel() for p in self.backbone.img_proj.parameters() if p.requires_grad),
                "qpos_proj": sum(p.numel() for p in self.qpos_proj.parameters() if p.requires_grad),
                "cvae": sum(p.numel() for p in self.cvae.parameters() if p.requires_grad),
                "policy_decoder": sum(p.numel() for p in self.policy_decoder.parameters() if p.requires_grad),
                "point_3d_encoder": sum(p.numel() for p in self.point_3d_encoder.parameters() if p.requires_grad) if self.point_3d_encoder is not None else 0,
                "use_3d_features": self.use_3d_features,
            }
        }
        return summary
