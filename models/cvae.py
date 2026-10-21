#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVAE (Conditional Variational Autoencoder) Module for ACT
- Encodes future actions chunk [50, 16] + current qpos [16] into a latent distribution z ~ N(mu, sigma^2)
- In Training: Reparameterization trick z = mu + eps * sigma
- In Eval / Inference: z = 0 (Fixed zero vector)
- Projects z (latent_dim=32) back to d_model (512) as a memory token for policy decoder
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEEncoder(nn.Module):
    """
    CVAE Transformer Encoder:
    Nén chuỗi hành động tương lai (k bước) cùng góc khớp hiện tại (qpos)
    thành phân phối Gaussian đa chiều trong không gian ẩn z in R^{latent_dim}.
    """
    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_dim: int = 16,
        qpos_dim: int = 16,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim

        # Projections
        self.action_proj = nn.Linear(action_dim, d_model)
        self.qpos_proj = nn.Linear(qpos_dim, d_model)

        # [CLS] Token học được đại diện cho toàn bộ phân phối
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer Encoder 2 tầng
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent Heads chiếu từ d_model -> latent_dim (32)
        self.latent_mu = nn.Linear(d_model, latent_dim)
        self.latent_logvar = nn.Linear(d_model, latent_dim)
        
        # Latent Projection chiếu z (32) ngược lại d_model (512) để làm token bộ nhớ
        self.latent_proj = nn.Linear(latent_dim, d_model)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick: z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Inputs:
            qpos: [B, 16] - Góc khớp 16 motor hiện tại
            actions: [B, chunk_size, 16] (hoặc None khi Inference)
            is_pad: [B, chunk_size] bool (True nếu là bước đệm padding)
        Outputs:
            z_token: [B, 1, d_model] - Token biểu diễn phong cách thao tác
            mu: [B, latent_dim] (None khi inference)
            logvar: [B, latent_dim] (None khi inference)
        """
        B = qpos.shape[0]
        device = qpos.device

        if actions is not None:
            # 1. Chế độ HUẤN LUYỆN (Training): Tính toán phân phối q(z | actions, qpos)
            action_tokens = self.action_proj(actions)    # [B, chunk_size, d_model]
            qpos_token = self.qpos_proj(qpos).unsqueeze(1) # [B, 1, d_model]
            cls_tokens = self.cls_token.repeat(B, 1, 1) # [B, 1, d_model]

            # Ghép chuỗi: [CLS, qpos, a_0, a_1, ..., a_{k-1}] -> Độ dài: 1 + 1 + chunk_size
            enc_input = torch.cat([cls_tokens, qpos_token, action_tokens], dim=1)

            # Xử lý padding mask nếu có
            src_key_padding_mask = None
            if is_pad is not None:
                # cls_token và qpos_token không bao giờ bị pad (False)
                prefix_mask = torch.zeros(B, 2, dtype=torch.bool, device=device)
                src_key_padding_mask = torch.cat([prefix_mask, is_pad], dim=1) # [B, 2 + chunk_size]

            # Đưa qua 2 tầng Transformer Encoder
            enc_out = self.encoder(enc_input, src_key_padding_mask=src_key_padding_mask)

            # Trích xuất đầu ra tại vị trí [CLS] token (vị trí 0)
            cls_out = enc_out[:, 0]  # [B, d_model]

            mu = self.latent_mu(cls_out)         # [B, latent_dim]
            logvar = self.latent_logvar(cls_out) # [B, latent_dim]

            # Lấy mẫu ngẫu nhiên z qua reparameterization trick
            z = self.reparameterize(mu, logvar)  # [B, latent_dim]
        else:
            # 2. Chế độ SUY LUẬN / ĐÁNH GIÁ (Eval / Inference):
            # Theo chuẩn ACT: gán cứng z = 0 (phân phối tiên nghiệm chuẩn N(0, I) tại mean)
            z = torch.zeros(B, self.latent_dim, device=device)
            mu, logvar = None, None

        # Chiếu z lên d_model thành 1 token bộ nhớ
        z_token = self.latent_proj(z).unsqueeze(1) # [B, 1, d_model]
        return z_token, mu, logvar
