#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Export Utilities for ACT (Action Chunking with Transformers)
Bimanual OpenArm (16-DOF: 2 arms x 8 motors) + Chest RGB-D Camera

Provides:
1. ACTInferenceWrapper: Clean inference-only forward module (image, qpos) -> action_chunk
2. export_act_to_onnx: Full export function with metadata embedding and ONNX Runtime validation
3. ONNXModelSessionWrapper: Drop-in wrapper to run inference on an exported .onnx model
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ACTInferenceWrapper(nn.Module):
    """
    Inference wrapper for ACT Policy.
    Exposes a clean forward interface for ONNX tracing:
      Inputs:
        - image: [B, in_channels, H, W] (Float32 normalized chest RGB-D)
        - qpos:  [B, qpos_dim]          (Float32 normalized current 16 joint angles)
      Outputs:
        - action_chunk: [B, chunk_size, action_dim] (Predicted normalized trajectory)
    """
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy.module if hasattr(policy, "module") else policy

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        return self.policy.predict_action_chunk(image, qpos)


def export_act_to_onnx(
    model: nn.Module,
    output_path: str,
    config: Optional[Any] = None,
    stats: Optional[Dict[str, Any]] = None,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verify_with_ort: bool = True,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Export an ACT Policy model to ONNX format.

    Args:
        model: Trained ACTPolicy model (or nn.DataParallel)
        output_path: Destination path for .onnx file
        config: ModelConfig instance or dict of hyperparameters
        stats: Normalization stats dict (action_mean, action_std, qpos_mean, qpos_std)
        opset_version: ONNX opset version (18 recommended for PyTorch 2.x)
        dynamic_batch: Whether to allow variable batch size (e.g. 1, 2, 4...)
        verify_with_ort: Verify exported model with onnxruntime
        verbose: Print detailed graph export steps
        logger: Optional logger instance for logging progress

    Returns:
        output_path: Absolute path to the exported ONNX model
    """
    def log_info(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def log_warn(msg: str):
        if logger:
            logger.warning(msg)
        else:
            print(f"[!] {msg}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Unwrap model and create CPU inference wrapper for clean export
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    # Determine input dimensions from config or model attributes (Chuẩn 480x640)
    img_height = getattr(raw_model.config, "img_height", 480) if hasattr(raw_model, "config") else 480
    img_width = getattr(raw_model.config, "img_width", 640) if hasattr(raw_model, "config") else 640
    in_channels = getattr(raw_model.config, "in_channels", 4) if hasattr(raw_model, "config") else 4
    num_cameras = getattr(raw_model.config, "num_cameras", 2) if hasattr(raw_model, "config") else 2
    camera_names = list(getattr(raw_model.config, "camera_names", ["chest", "front"])) if hasattr(raw_model, "config") else ["chest", "front"]
    qpos_dim = getattr(raw_model.config, "qpos_dim", 16) if hasattr(raw_model, "config") else 16
    chunk_size = getattr(raw_model.config, "chunk_size", 50) if hasattr(raw_model, "config") else 50
    action_dim = getattr(raw_model.config, "action_dim", 16) if hasattr(raw_model, "config") else 16

    wrapper = ACTInferenceWrapper(raw_model).eval()

    # Create dummy inputs on the same device as model parameters
    param_device = next(raw_model.parameters()).device
    # Use batch size = 2 for dummy inputs when dynamic_batch=True to avoid Dim(1) specialization guards
    sample_batch = 2 if dynamic_batch else 1
    if num_cameras > 1:
        dummy_image = torch.randn(sample_batch, num_cameras, in_channels, img_height, img_width, device=param_device)
        img_shape_str = f"[{sample_batch}, {num_cameras}, {in_channels}, {img_height}, {img_width}]"
    else:
        dummy_image = torch.randn(sample_batch, in_channels, img_height, img_width, device=param_device)
        img_shape_str = f"[{sample_batch}, {in_channels}, {img_height}, {img_width}]"
    dummy_qpos = torch.randn(sample_batch, qpos_dim, device=param_device)

    log_info(f"[*] Đang xuất mô hình ACT sang ONNX tại: {output_path}")
    log_info(f"    - Kích thước đầu vào: image {img_shape_str}, qpos [{sample_batch}, {qpos_dim}]")
    log_info(f"    - Kích thước đầu ra: action_chunk [{sample_batch}, {chunk_size}, {action_dim}]")
    log_info(f"    - ONNX Opset Version: {opset_version}")
    log_info(f"    - Dynamic Batch Size : {'BẬT' if dynamic_batch else 'TẮT (Batch=1)'}")

    # Setup dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "qpos": {0: "batch"},
            "action_chunk": {0: "batch"},
        }

    # Disable PyTorch 2.x _native_multi_head_attention fastpath for clean ONNX export
    prev_fastpath = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        try:
            prev_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            pass

    try:
        try:
            torch.onnx.export(
                wrapper,
                (dummy_image, dummy_qpos),
                output_path,
                input_names=["image", "qpos"],
                output_names=["action_chunk"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                verbose=verbose,
            )
        except Exception as exc:
            log_warn(f"Xuất ONNX với dynamic_axes gặp lỗi: {exc}. Đang thử lại với batch cố định...")
            torch.onnx.export(
                wrapper,
                (dummy_image[:1], dummy_qpos[:1]),
                output_path,
                input_names=["image", "qpos"],
                output_names=["action_chunk"],
                opset_version=opset_version,
                verbose=verbose,
            )
    finally:
        if prev_fastpath is not None:
            try:
                torch.backends.mha.set_fastpath_enabled(prev_fastpath)
            except Exception:
                pass

    # Prepare metadata dictionary
    meta_dict: Dict[str, Any] = {
        "format": "ACT_Bimanual_OpenArm_ONNX",
        "version": "1.0.0",
        "input_names": ["image", "qpos"],
        "output_names": ["action_chunk"],
        "in_channels": in_channels,
        "num_cameras": num_cameras,
        "camera_names": camera_names,
        "img_height": img_height,
        "img_width": img_width,
        "qpos_dim": qpos_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
    }

    if config is not None:
        if hasattr(config, "__dict__"):
            meta_dict["config"] = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        elif isinstance(config, dict):
            meta_dict["config"] = config

    if stats is not None:
        serializable_stats = {}
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                serializable_stats[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                serializable_stats[k] = v.cpu().tolist()
            else:
                serializable_stats[k] = v
        meta_dict["normalization_stats"] = serializable_stats

    # Save companion JSON metadata file
    meta_json_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    # Embed metadata into ONNX model and make model self-contained (no .data file)
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(output_path, load_external_data=True)
            meta_str = json.dumps(meta_dict, ensure_ascii=False)
            meta_entry = onnx_model.metadata_props.add()
            meta_entry.key = "act_metadata"
            meta_entry.value = meta_str
            # Lưu gọn thành 1 file .onnx duy nhất, không tách file .onnx.data
            onnx.save_model(onnx_model, output_path, save_as_external_data=False)
            ext_data_file = output_path + ".data"
            if os.path.exists(ext_data_file):
                os.remove(ext_data_file)
        except Exception as exc:
            log_warn(f"Không thể nhúng metadata trực tiếp vào file ONNX: {exc}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log_info(f"[✓] Đã xuất thành công mô hình ONNX: {output_path} ({file_size_mb:.2f} MB)")
    log_info(f"[✓] Đã lưu metadata đi kèm: {meta_json_path}")

    # Verify with ONNX Runtime if requested
    if verify_with_ort and ONNX_AVAILABLE:
        try:
            log_info("[*] Đang kiểm tra tính tương thích và độ chính xác số học với ONNX Runtime...")
            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            
            # Test batch size = 1
            test_img = dummy_image[:1].cpu().numpy()
            test_qpos = dummy_qpos[:1].cpu().numpy()
            ort_inputs = {"image": test_img, "qpos": test_qpos}
            ort_outs = session.run(["action_chunk"], ort_inputs)[0]

            with torch.no_grad():
                torch_out = wrapper(dummy_image[:1], dummy_qpos[:1]).cpu().numpy()

            max_diff = float(np.max(np.abs(ort_outs - torch_out)))
            log_info(f"[✓] Kiểm tra ONNX Runtime thành công! Output shape: {ort_outs.shape}")
            log_info(f"    - Sai lệch lớn nhất so với PyTorch (Max Diff): {max_diff:.6e}")
            if max_diff > 1e-3:
                log_warn(f"Sai lệch ONNX Runtime hơi lớn ({max_diff:.4f}). Vui lòng kiểm tra lại opset.")
            else:
                log_info("    - Kết quả số học khớp hoàn hảo giữa PyTorch và ONNX Runtime (✓)")
        except Exception as exc:
            log_warn(f"Kiểm tra ONNX Runtime thất bại: {exc}")

    return os.path.abspath(output_path)


class ONNXModelSessionWrapper:
    """
    Convenient wrapper to run inference using ONNX Runtime.
    Mimics ACTPolicy interface: model(image=..., qpos=..., actions=None)
    """
    def __init__(self, onnx_path: str, device: str = "cpu"):
        if not ONNX_AVAILABLE:
            raise ImportError("Vui lòng cài đặt onnx và onnxruntime: pip install onnx onnxruntime")

        self.onnx_path = onnx_path
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Load companion metadata if available
        meta_path = os.path.splitext(onnx_path)[0] + "_metadata.json"
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        qpos: Union[torch.Tensor, np.ndarray],
        actions: Optional[Any] = None,
        is_pad: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Runs inference on image and qpos.
        Returns:
            pred_actions: torch.Tensor of shape [B, chunk_size, action_dim]
            mu: None
            logvar: None
        """
        # Convert torch tensors to numpy if needed
        is_torch = isinstance(image, torch.Tensor)
        device = image.device if is_torch else torch.device("cpu")

        img_np = image.detach().cpu().numpy() if is_torch else np.asarray(image, dtype=np.float32)
        qpos_np = qpos.detach().cpu().numpy() if isinstance(qpos, torch.Tensor) else np.asarray(qpos, dtype=np.float32)

        # Run ONNX Runtime
        ort_outs = self.session.run(["action_chunk"], {"image": img_np, "qpos": qpos_np})[0]

        pred_actions = torch.from_numpy(ort_outs).to(device)
        return pred_actions, None, None

    def predict_action_chunk(
        self,
        image: Union[torch.Tensor, np.ndarray],
        qpos: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        pred_actions, _, _ = self(image=image, qpos=qpos)
        return pred_actions

    def eval(self):
        return self

    def train(self, mode: bool = False):
        return self

