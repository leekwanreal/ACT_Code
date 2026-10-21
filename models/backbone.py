#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Backbone Modules for ACT:
1. RGBDYOLOBackbone: YOLOv8 CSPDarknet + SPPF (SOTA Real-Time Object Detection)
   - Tối ưu hóa đặc biệt cho bài toán gắp nhặt đồ (Pick-and-Place) với khả năng định vị biên & tâm vật thể.
   - Nhận ảnh 4 kênh RGB-D [B, 4, H, W].
   - Tích hợp khối SPPF (Spatial Pyramid Pooling: 5x5, 9x9, 13x13) bắt đặc trưng vật thể đa tỉ lệ.
   - 1x1 Conv Image Projection chiếu về d_model (512).
   - 2D Sinusoidal Positional Encoding (Fixed).
2. RGBDResNetBackbone: ResNet-18 / ResNet-34 / ResNet-50 4 Kênh RGB-D (Chuẩn Stanford ALOHA ACT).
   - Nhận ảnh 4 kênh RGB-D [B, 4, H, W].
   - Khởi tạo Stem Adapter 4 kênh với trọng số RGB từ ImageNet, kênh Depth từ trung bình RGB.
   - Selective Freezing: Đóng băng các tầng sâu, mở khóa conv1_adapter để học dung hợp độ sâu.
   - Hỗ trợ Stride 16 (1200 tokens/cam) hoặc Stride 32 (300 tokens/cam).
3. RGBDConvNeXtBackbone: ConvNeXt-Nano (Pure-CNN SOTA spatial feature extractor).
"""

import os
import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 2D Sinusoidal Positional Encoding
# ==============================================================================

class SinusoidalPositionEmbedding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding cố định (không học - Fixed).
    Mã hóa tọa độ không gian 2D cho từng patch đặc trưng theo công thức DETR/ACT.
    """
    def __init__(self, d_model: int = 512, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model ({d_model}) phải chia hết cho 4 để chia đều cho sin/cos x và y."
        self.d_model = d_model
        self.temperature = temperature
        self.num_pos_feats = d_model // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: pos: [B, d_model, H, W]
        """
        B, _, H, W = x.shape
        device = x.device
        
        # Tạo lưới tọa độ 1..H và 1..W
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # Ghép [H, W, d_model] -> Permute [d_model, H, W] -> Repeat cho Batch [B, d_model, H, W]
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


# ==============================================================================
# 2. Các khối cấu trúc cơ bản của YOLOv8 (PyTorch thuần túy)
# ==============================================================================

def autopad(k, p=None, d=1):
    """Tính toán padding tự động để giữ nguyên kích thước không gian."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Khối Bottleneck chuẩn với kết nối tắt residual."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, c_, k[0], 1)
        self.cv2 = ConvBNSiLU(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck với 2 tích chập (Cross Stage Partial with 2 Convolutions).
    Tối ưu hóa mạnh mẽ dòng gradient cho bài toán phát hiện vật thể.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF).
    Gom cụm đặc trưng đa tỉ lệ (5x5, 9x9, 13x13) giúp định vị vật thể ở các khoảng cách khác nhau trên mặt bàn.
    """
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# ==============================================================================
# 3. Vision Backbone YOLOv8 4 Kênh RGB-D (RGBDYOLOBackbone)
# ==============================================================================

class RGBDYOLOBackbone(nn.Module):
    """
    Vision Backbone YOLOv8 CSPDarknet + SPPF cho ACT:
    - Đầu vào: [B, 4, H, W] (3 kênh RGB + 1 kênh Depth Z đã chuẩn hóa).
    - Khối Stem 4 kênh: Tự động học cách hòa trộn chiều sâu với màu sắc.
    - Khối C2f & SPPF: Bắt đặc trưng vật thể đa tỉ lệ cực nhạy cho bài toán gắp nhặt.
    - 1x1 Conv Image Projection: Chiếu từ C_out của YOLO về d_model (512).
    - 2D Sinusoidal PE: Mã hóa tọa độ không gian 2D cho từng patch.
    - Đầu ra: visual_tokens [B, N_tokens, d_model] khớp 100% với Policy Decoder của ACT.
    """
    def __init__(
        self,
        d_model: int = 512,
        freeze_backbone: bool = True,
        in_channels: int = 4,
        model_name: str = "yolov8_nano",
        pretrained: bool = True,
        stride: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.model_name = model_name
        self.stride = stride
        self.freeze_backbone = freeze_backbone

        # Cấu hình số kênh theo phiên bản nano (n) hoặc small (s)
        if "small" in model_name:
            # YOLOv8-Small
            c_stem = 32
            c_stages = [64, 128, 256, 512]
            n_blocks = [1, 2, 2, 1]
        else:
            # YOLOv8-Nano (Mặc định - Siêu nhẹ ~2.5M params, latency ~2ms)
            c_stem = 16
            c_stages = [32, 64, 128, 256]
            n_blocks = [1, 2, 2, 1]

        # 1. Stem Conv nhận 4 kênh [B, 4, H, W] -> Stride 2
        self.stem = ConvBNSiLU(in_channels, c_stem, k=3, s=2)

        # 2. Stage 1 (Stride 4)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c_stem, c_stages[0], k=3, s=2),
            C2f(c_stages[0], c_stages[0], n=n_blocks[0], shortcut=True),
        )

        # 3. Stage 2 (Stride 8)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c_stages[0], c_stages[1], k=3, s=2),
            C2f(c_stages[1], c_stages[1], n=n_blocks[1], shortcut=True),
        )

        # 4. Stage 3 (Stride 16 - Mức P4)
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c_stages[1], c_stages[2], k=3, s=2),
            C2f(c_stages[2], c_stages[2], n=n_blocks[2], shortcut=True),
        )

        if stride == 32:
            # Mức P5 (Stride 32 + SPPF)
            self.stage4 = nn.Sequential(
                ConvBNSiLU(c_stages[2], c_stages[3], k=3, s=2),
                C2f(c_stages[3], c_stages[3], n=n_blocks[3], shortcut=True),
            )
            self.sppf = SPPF(c_stages[3], c_stages[3], k=5)
            self.out_channels = c_stages[3]
        else:
            # Mức P4 (Stride 16 + SPPF) - KHUYÊN DÙNG CHO NHẶT ĐỒ (1200 tokens/camera với ảnh 480x640)
            self.stage4 = None
            self.sppf = SPPF(c_stages[2], c_stages[2], k=5)
            self.out_channels = c_stages[2]

        # Nạp trọng số COCO tiền huấn luyện nếu được yêu cầu
        if pretrained:
            self._init_pretrained_weights()

        # Đóng băng các tầng C2f/SPPF nếu freeze_backbone = True
        if freeze_backbone:
            # Đóng băng các tầng trích xuất đặc trưng COCO
            for stage in [self.stage1, self.stage2, self.stage3, self.sppf]:
                if stage is not None:
                    for p in stage.parameters():
                        p.requires_grad = False
                    stage.eval()
            if self.stage4 is not None:
                for p in self.stage4.parameters():
                    p.requires_grad = False
                self.stage4.eval()
            # Riêng Stem 4-kênh được giữ requires_grad = True để học đọc kênh Depth!
            for p in self.stem.parameters():
                p.requires_grad = True

        # 5. 1x1 Conv Image Projection chiếu về d_model (512) (Trainable)
        self.img_proj = nn.Conv2d(self.out_channels, d_model, kernel_size=1)

        # 6. 2D Sinusoidal Positional Encoding (Fixed)
        self.pos_embed_2d = SinusoidalPositionEmbedding2D(d_model=d_model)

    def _init_pretrained_weights(self):
        """
        Nạp trọng số tiền huấn luyện từ COCO cho YOLOv8:
        - Tự động tải yolov8n.pt / yolov8s.pt từ GitHub releases
        - Mock ultralytics classes để nạp an toàn bằng pure PyTorch
        - Ánh xạ 3 kênh RGB vào Stem Conv, khởi tạo kênh thứ 4 (Depth) từ trung bình 3 kênh RGB
        - Nạp toàn bộ các block C2f, Bottleneck và Conv của Stage 1, 2, 3 (và 4, SPPF nếu stride=32).
        """
        import sys
        import types

        # Tạm thời đăng ký mock modules cho ultralytics để torch.load deserialization thành công
        mock_mods = [
            'ultralytics', 'ultralytics.nn', 'ultralytics.nn.tasks',
            'ultralytics.nn.modules', 'ultralytics.nn.modules.conv',
            'ultralytics.nn.modules.block', 'ultralytics.nn.modules.head'
        ]
        registered_mocks = []
        for mod in mock_mods:
            if mod not in sys.modules:
                class MockModule(types.ModuleType):
                    def __getattr__(self, name):
                        class Dummy:
                            def __init__(self, *args, **kwargs): pass
                            def __setstate__(self, state):
                                if isinstance(state, dict): self.__dict__.update(state)
                        Dummy.__name__ = name
                        return Dummy
                sys.modules[mod] = MockModule(mod)
                registered_mocks.append(mod)

        try:
            is_nano = "nano" in self.model_name
            weights_url = (
                "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
                if is_nano else
                "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
            )
            cached_path = torch.hub.load_state_dict_from_url(
                weights_url,
                map_location="cpu",
                progress=False,
                check_hash=False,
                weights_only=False,
            )

            # Hàm đệ quy bóc tách state_dict từ các đối tượng module
            def _extract_sd(obj, prefix=""):
                sd = {}
                if hasattr(obj, "_parameters"):
                    for k, v in obj._parameters.items():
                        if v is not None: sd[f"{prefix}{k}"] = v
                if hasattr(obj, "_buffers"):
                    for k, v in obj._buffers.items():
                        if v is not None: sd[f"{prefix}{k}"] = v
                if hasattr(obj, "_modules"):
                    for k, v in obj._modules.items():
                        if v is not None: sd.update(_extract_sd(v, f"{prefix}{k}."))
                return sd

            if isinstance(cached_path, dict) and "model" in cached_path:
                raw_sd = _extract_sd(cached_path["model"])
            elif isinstance(cached_path, dict):
                raw_sd = cached_path
            else:
                raw_sd = _extract_sd(cached_path)

            mapping = {
                "stem": "model.0.",
                "stage1.0": "model.1.",
                "stage1.1": "model.2.",
                "stage2.0": "model.3.",
                "stage2.1": "model.4.",
                "stage3.0": "model.5.",
                "stage3.1": "model.6.",
            }
            if self.stage4 is not None:
                mapping["stage4.0"] = "model.7."
                mapping["stage4.1"] = "model.8."
                mapping["sppf"] = "model.9."

            matched = 0
            curr_sd = self.state_dict()
            for my_prefix, ultr_prefix in mapping.items():
                for k, v in raw_sd.items():
                    if k.startswith(ultr_prefix):
                        suffix = k[len(ultr_prefix):]
                        my_key = f"{my_prefix}.{suffix}"
                        if my_key in curr_sd and curr_sd[my_key].shape == v.shape:
                            curr_sd[my_key].copy_(v)
                            matched += 1
                        elif my_key == "stem.conv.weight":
                            w_4ch = torch.zeros_like(self.stem.conv.weight)
                            w_4ch[:, :3, :, :] = v
                            w_4ch[:, 3:4, :, :] = v.mean(dim=1, keepdim=True)
                            self.stem.conv.weight.data.copy_(w_4ch)
                            matched += 1

            print(f"[COCO Weights] Successfully loaded {matched} layers from COCO pre-trained weights for {self.model_name}.")
        except Exception as e:
            print(f"[Notice] Could not load online COCO weights ({e}). Initializing backbone with Kaiming Normal.")
        finally:
            # Dọn dẹp mock modules để tránh side effect
            for mod in registered_mocks:
                sys.modules.pop(mod, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, 4, H, W] - Ảnh 4 kênh RGB-D đã chuẩn hóa
        Output:
            visual_tokens: [B, N_tokens, d_model]
        """
        # 1. Trích xuất qua các tầng CSPDarknet
        feat = self.stem(x)       # [B, 16, H/2, W/2]
        feat = self.stage1(feat)   # [B, 32, H/4, W/4]
        feat = self.stage2(feat)   # [B, 64, H/8, W/8]
        feat = self.stage3(feat)   # [B, 128, H/16, W/16]

        if self.stage4 is not None:
            feat = self.stage4(feat)

        feat = self.sppf(feat)     # [B, C_out, H_feat, W_feat]

        # 2. Chiếu về d_model và thêm mã hóa vị trí 2D
        proj_feat = self.img_proj(feat)           # [B, d_model, H_feat, W_feat]
        pos = self.pos_embed_2d(proj_feat)         # [B, d_model, H_feat, W_feat]
        feat_with_pos = proj_feat + pos

        # 3. Làm phẳng thành chuỗi Visual Tokens cho Transformer Decoder: [B, N_tokens, d_model]
        visual_tokens = feat_with_pos.flatten(2).permute(0, 2, 1)
        return visual_tokens


# ==============================================================================
# 4. Vision Backbone ConvNeXt-Nano 4 Kênh (RGBDConvNeXtBackbone)
# ==============================================================================

class RGBDConvNeXtBackbone(nn.Module):
    """
    Khối thị giác RGB-D sử dụng ConvNeXt-Nano cho mô hình ACT:
    - Nhận vào ảnh 4 kênh [B, 4, H, W] (3 RGB + 1 Depth).
    - Sử dụng ConvNeXt-Nano (timm) với in_chans=4, kernel lớn 7x7 và Inverted Bottleneck.
    - 1x1 Conv Image Projection: Chiếu 640 kênh đặc trưng về d_model (512).
    - 2D Sinusoidal PE: Mã hóa tọa độ không gian 2D cho từng patch.
    - Đầu ra: visual_tokens [B, N_patches, d_model] ([B, 300, 512] với ảnh 480x640).
    """
    def __init__(
        self,
        d_model: int = 512,
        freeze_backbone: bool = True,
        in_channels: int = 4,
        model_name: str = "convnext_nano",
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.model_name = model_name

        # Kiểm tra bắt buộc thư viện timm
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "[LỖI BẮT BUỘC] Không tìm thấy thư viện 'timm' để khởi tạo ConvNeXt-Nano! "
                "Cài đặt: pip install timm>=1.0.0"
            ) from exc

        # Khởi tạo ConvNeXt-Nano từ timm
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
            )
            in_features = self.backbone.num_features
        except Exception as exc:
            raise RuntimeError(
                f"[LỖI BẮT BUỘC] Không thể tải/khởi tạo mô hình '{model_name}' với {in_channels} kênh: {exc}."
            ) from exc

        # Đóng băng backbone nếu freeze_backbone = True
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 1x1 Conv Image Projection (Trainable)
        self.img_proj = nn.Conv2d(in_features, d_model, kernel_size=1)

        # 2D Sinusoidal Positional Encoding (Fixed)
        self.pos_embed_2d = SinusoidalPositionEmbedding2D(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, 4, H, W] - Ảnh 4 kênh RGB-D đã chuẩn hóa
        Output:
            visual_tokens: [B, N_patches, d_model]
        """
        feat = self.backbone.forward_features(x)  # [B, 640, H_feat, W_feat]
        proj_feat = self.img_proj(feat)           # [B, d_model, H_feat, W_feat]
        pos = self.pos_embed_2d(proj_feat)         # [B, d_model, H_feat, W_feat]
        feat_with_pos = proj_feat + pos
        visual_tokens = feat_with_pos.flatten(2).permute(0, 2, 1)
        return visual_tokens


# ==============================================================================
# 5. Vision Backbone ResNet 4 Kênh RGB-D (RGBDResNetBackbone)
# ==============================================================================

class RGBDResNetBackbone(nn.Module):
    """
    Vision Backbone ResNet (ResNet-18 / ResNet-34 / ResNet-50) 4 Kênh RGB-D cho ACT:
    - Đầu vào: [B, 4, H, W] (3 kênh RGB chuẩn hóa ImageNet + 1 kênh Depth chuẩn hóa [0, 1]).
    - Khối Conv1 Adapter 4 kênh: Khởi tạo 3 kênh RGB từ pre-trained weights (ImageNet),
      kênh thứ 4 (Depth) khởi tạo bằng trung bình 3 kênh RGB: W_depth = 1/3 * sum(W_c).
    - Selective Freezing: Đóng băng các tầng trích xuất sâu (bn1, layer1..4) khi freeze_backbone=True,
      giữ conv1_adapter mở khóa huấn luyện (requires_grad=True) với lr_backbone nhỏ (1e-5).
    - Hỗ trợ Stride 16 (1200 visual tokens/camera @ 480x640) hoặc Stride 32 (300 visual tokens/camera).
    - 1x1 Conv Image Projection: Chiếu từ out_channels (512 / 2048) về d_model (512).
    - 2D Sinusoidal Positional Encoding (Fixed).
    - Đầu ra: visual_tokens [B, N_tokens, d_model] tương thích 100% với Policy Decoder của ACT.
    """
    def __init__(
        self,
        d_model: int = 512,
        freeze_backbone: bool = True,
        in_channels: int = 4,
        model_name: str = "resnet18",
        pretrained: bool = True,
        stride: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.model_name = "resnet18" if model_name == "resnet" else model_name
        self.stride = stride
        self.freeze_backbone = freeze_backbone

        # 1. Khởi tạo mạng ResNet nền tảng từ torchvision
        import torchvision.models as tv_models

        if self.model_name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet18(weights=weights)
            self.out_channels = 512
        elif self.model_name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet34(weights=weights)
            self.out_channels = 512
        elif self.model_name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = tv_models.resnet50(weights=weights)
            self.out_channels = 2048
        else:
            raise ValueError(
                f"[LỖI] Biến thể ResNet '{model_name}' không được hỗ trợ. "
                f"Vui lòng chọn 'resnet18', 'resnet34' hoặc 'resnet50'."
            )

        # 2. Thay thế Conv1 bằng Conv 4-kênh (conv1_adapter)
        orig_conv1 = resnet.conv1
        self.conv1_adapter = nn.Conv2d(
            in_channels,
            orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False,
        )

        # Nạp trọng số ImageNet cho conv1 4 kênh
        if pretrained and hasattr(orig_conv1, "weight") and orig_conv1.weight is not None:
            with torch.no_grad():
                w_orig = orig_conv1.weight.data
                w_4ch = torch.zeros_like(self.conv1_adapter.weight.data)
                w_4ch[:, :3, :, :] = w_orig[:, :3, :, :] if w_orig.shape[1] >= 3 else w_orig
                w_4ch[:, 3:4, :, :] = w_orig[:, :3, :, :].mean(dim=1, keepdim=True)
                self.conv1_adapter.weight.data.copy_(w_4ch)
            print(f"[ResNet Weights] Successfully initialized conv1_adapter (4 channels) from ImageNet pre-trained weights ({self.model_name}).")

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 3. Điều chỉnh Stride cho layer4 nếu chọn stride=16 (mặc định)
        if self.stride == 16:
            first_block = self.layer4[0]
            if isinstance(first_block, tv_models.resnet.BasicBlock):
                first_block.conv1.stride = (1, 1)
            elif isinstance(first_block, tv_models.resnet.Bottleneck):
                first_block.conv2.stride = (1, 1)
            elif hasattr(first_block, "conv1") and first_block.conv1.stride == (2, 2):
                first_block.conv1.stride = (1, 1)
            elif hasattr(first_block, "conv2") and first_block.conv2.stride == (2, 2):
                first_block.conv2.stride = (1, 1)

            if first_block.downsample is not None and hasattr(first_block.downsample[0], "stride"):
                first_block.downsample[0].stride = (1, 1)

        # 4. Đóng băng backbone nếu freeze_backbone = True
        if freeze_backbone:
            for stage in [self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in stage.parameters():
                    p.requires_grad = False
                stage.eval()
            # conv1_adapter 4 kênh giữ requires_grad = True để học đọc Depth!
            for p in self.conv1_adapter.parameters():
                p.requires_grad = True

        # 5. 1x1 Conv Image Projection chiếu về d_model (512) (Trainable)
        self.img_proj = nn.Conv2d(self.out_channels, d_model, kernel_size=1)

        # 6. 2D Sinusoidal Positional Encoding (Fixed)
        self.pos_embed_2d = SinusoidalPositionEmbedding2D(d_model=d_model)

    @property
    def conv1(self):
        """Thuộc tính truy cập nhanh conv1_adapter để tương thích với API ResNet chuẩn."""
        return self.conv1_adapter

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone:
            # Đảm bảo các tầng đóng băng giữ nguyên eval mode để không lệch thống kê BatchNorm
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, 4, H, W] - Ảnh 4 kênh RGB-D đã chuẩn hóa
        Output:
            visual_tokens: [B, N_tokens, d_model]
        """
        feat = self.conv1_adapter(x)       # [B, 64, H/2, W/2]
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)          # [B, 64, H/4, W/4]

        feat = self.layer1(feat)           # [B, 64/256, H/4, W/4]
        feat = self.layer2(feat)           # [B, 128/512, H/8, W/8]
        feat = self.layer3(feat)           # [B, 256/1024, H/16, W/16]
        feat = self.layer4(feat)           # [B, 512/2048, H/stride, W/stride]

        # Chiếu về d_model và thêm mã hóa vị trí 2D
        proj_feat = self.img_proj(feat)    # [B, d_model, H_feat, W_feat]
        pos = self.pos_embed_2d(proj_feat)  # [B, d_model, H_feat, W_feat]
        feat_with_pos = proj_feat + pos

        # Làm phẳng thành chuỗi Visual Tokens cho Transformer Decoder: [B, N_tokens, d_model]
        visual_tokens = feat_with_pos.flatten(2).permute(0, 2, 1)
        return visual_tokens


# ==============================================================================
# 6. Factory khởi tạo Vision Backbone
# ==============================================================================

def build_vision_backbone(
    backbone_type: str = "yolov8_nano",
    d_model: int = 512,
    freeze_backbone: bool = True,
    in_channels: int = 4,
    stride: int = 16,
    yolo_stride: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory tạo Vision Backbone cho ACT.
    Hỗ trợ:
      - 'yolov8_nano': YOLOv8-Nano CSPDarknet + SPPF 4 kênh (Tối ưu cho nhặt đồ / object detection).
      - 'yolov8_small': YOLOv8-Small CSPDarknet + SPPF 4 kênh.
      - 'resnet18' / 'resnet': ResNet-18 4 kênh (RGB-D) chuẩn Stanford ALOHA ACT.
      - 'resnet34': ResNet-34 4 kênh (RGB-D).
      - 'resnet50': ResNet-50 4 kênh (RGB-D).
      - 'convnext_nano': ConvNeXt-Nano Pure-CNN.
    """
    if yolo_stride is not None:
        stride = yolo_stride

    if backbone_type.startswith("yolov8"):
        return RGBDYOLOBackbone(
            d_model=d_model,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            model_name=backbone_type,
            pretrained=True,
            stride=stride,
        )
    elif backbone_type.startswith("resnet") or backbone_type == "resnet":
        model_name = "resnet18" if backbone_type == "resnet" else backbone_type
        return RGBDResNetBackbone(
            d_model=d_model,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            model_name=model_name,
            pretrained=True,
            stride=stride,
        )
    elif backbone_type == "convnext_nano":
        return RGBDConvNeXtBackbone(
            d_model=d_model,
            freeze_backbone=freeze_backbone,
            in_channels=in_channels,
            model_name="convnext_nano",
            pretrained=True,
        )
    else:
        raise ValueError(
            f"[LỖI] Backbone '{backbone_type}' không được hỗ trợ. "
            f"Vui lòng chọn 'yolov8_nano', 'yolov8_small', 'resnet18', 'resnet34', 'resnet50' hoặc 'convnext_nano'."
        )

