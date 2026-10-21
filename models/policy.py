#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Policy Decoder Module for ACT
- Action Queries: k=50 learnable embeddings + 1D Sinusoidal Temporal PE
- Transformer Decoder (4 Layers) with Multi-Head Self-Attn & Cross-Attn over 302 Memory Tokens:
    [300 Visual Tokens + 1 qpos Token + 1 Latent z Token]
- Action Head: Linear 512 -> 16 Joint Targets
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding1D(nn.Module):
    """
    1D Sinusoidal Temporal Positional Encoding (Fixed)
    Mã hóa thứ tự thời gian cho chuỗi dự đoán 50 bước tương lai [t, t+1, ..., t+49].
    """
    def __init__(self, d_model: int = 512, max_len: int = 100, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(temperature) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, length: int) -> torch.Tensor:
        """
        Trả về PE cho độ dài length: [1, length, d_model]
        """
        return self.pe[:, :length, :]


class TransformerPolicyDecoder(nn.Module):
    """
    Transformer Policy Decoder cho ACT:
    - 50 Action Queries tự học biểu diễn ý định hành động trong tương lai.
    - 1D Temporal Sinusoidal PE cung cấp nhãn thời gian thứ tự 0..49.
    - 4 Layers Transformer Decoder với Multi-Head Self-Attention và Cross-Attention.
    - Memory tokens gồm 302 phần tử: [300 Visual + 1 qpos + 1 z].
    - Action Linear Head chiếu từ d_model (512) ra 16 góc mục tiêu cho 16 động cơ Damiao.
    """
    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 50,
        action_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 50 Action Queries học được (Learnable Parameter)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)

        # 1D Temporal Sinusoidal PE (Fixed)
        self.temporal_pe = SinusoidalPositionEmbedding1D(d_model=d_model, max_len=chunk_size + 10)

        # 4 tầng Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action Linear Head: 512 -> 16 góc mục tiêu
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            memory: [B, 302, d_model] - Memory tokens (300 Visual + 1 qpos + 1 z)
        Outputs:
            pred_actions: [B, chunk_size, action_dim] - Quỹ đạo 50 bước góc cho 16 động cơ
        """
        B = memory.shape[0]

        # 1. Khởi tạo Action Queries với Temporal PE
        queries = self.action_queries.repeat(B, 1, 1) # [B, chunk_size, d_model]
        pe = self.temporal_pe(self.chunk_size)        # [1, chunk_size, d_model]
        tgt = queries + pe                            # [B, chunk_size, d_model]

        # 2. Giải mã qua 4 tầng Transformer Decoder (Cross-Attention với Memory)
        decoded = self.decoder(tgt=tgt, memory=memory) # [B, chunk_size, d_model]

        # 3. Action Head chiếu ra 16 khớp
        pred_actions = self.action_head(decoded)      # [B, chunk_size, action_dim]
        return pred_actions
