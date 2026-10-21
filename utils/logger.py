#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Logging System for ACT Training & Evaluation
Provides:
1. Console Logger: Clean, structured terminal output with timestamps and status indicators.
2. File Logger: Comprehensive text log in log_dir/train.log.
3. JSONL Logger: Machine-readable metric tracking in log_dir/metrics.jsonl for plotting.
4. TensorBoard Logger: Real-time visual monitoring of loss curves, LR, and validation metrics.
5. Optional WandB Logger: Cloud tracking integration if enabled.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricLogger:
    """
    Trình quản lý ghi chép (Logger) toàn diện cho quá trình Huấn luyện & Đánh giá ACT.
    """
    def __init__(
        self,
        log_dir: str,
        exp_name: str = "act_experiment",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "act-openarm",
        wandb_entity: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        save_log_file: bool = False,
        save_metrics_jsonl: bool = False,
        save_config_json: bool = False,
    ):
        if not log_dir:
            raise ValueError("log_dir không được rỗng hoặc None.")

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.exp_name = exp_name
        self.save_log_file = save_log_file
        self.save_metrics_jsonl = save_metrics_jsonl
        self.save_config_json = save_config_json
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        self.log_file = os.path.join(self.log_dir, "train.log")

        # 1. Thiết lập File & Stream Logger
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # File Handler (chỉ tạo khi save_log_file=True)
        if self.save_log_file:
            file_handler = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
            file_formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Console Stream Handler (luôn luôn ghi ra Terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 2. Khởi tạo TensorBoard — STRICT: nếu yêu cầu mà không có thì PHẢI báo lỗi
        self.tb_writer = None
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError(
                    "[LỖI NGHIÊM TRỌNG] use_tensorboard=True nhưng package 'tensorboard' chưa được cài đặt! "
                    "Chạy: pip install tensorboard"
                )
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            if self.tb_writer is None:
                raise RuntimeError(
                    f"[LỖI NGHIÊM TRỌNG] Không thể khởi tạo SummaryWriter tại: {tb_dir}"
                )
            self.info(f"[✓] Đã khởi tạo TensorBoard tại: {tb_dir}")

        # 3. Khởi tạo WandB (nếu được yêu cầu và cài đặt)
        self.use_wandb = use_wandb
        if self.use_wandb:
            try:
                import wandb
                # Tự động đăng nhập nếu có API key
                api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
                if api_key:
                    wandb.login(key=api_key)
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=exp_name,
                    config=config_dict,
                )
                wandb.define_metric("train/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("epoch")
                wandb.define_metric("train_epoch/*", step_metric="epoch")
                wandb.define_metric("val_epoch/*", step_metric="epoch")
                self.info(f"[✓] Đã kết nối với Weights & Biases (Project: {wandb_project}, Run: {exp_name}).")
            except Exception as e:
                self.error(f"[LỖI] Không thể khởi động WandB: {e}")
                raise

        # Lưu config dưới dạng JSON (chỉ khi save_config_json=True)
        if self.save_config_json and config_dict is not None:
            config_path = os.path.join(self.log_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)
            self.info(f"[✓] Đã lưu cấu hình tại: {config_path}")

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def _require_tb_writer(self, caller: str) -> bool:
        """Kiểm tra bắt buộc: TensorBoard writer phải sẵn sàng nếu use_tensorboard=True."""
        if not self.use_tensorboard:
            return False  # Người dùng chủ động tắt, không ghi nhưng không lỗi
        if self.tb_writer is None:
            raise RuntimeError(
                f"[LỖI] {caller}: use_tensorboard=True nhưng tb_writer=None. "
                f"TensorBoard đã bị lỗi trong quá trình khởi tạo."
            )
        return True

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Ghi nhận các độ đo metrics vào:
        1. JSONL file (máy đọc)
        2. TensorBoard (đồ thị trực quan)
        3. WandB (nếu có)
        """
        if metrics is None:
            self.error("[LỖI] log_metrics(): metrics không được là None.")
            raise ValueError("log_metrics(): metrics không được là None.")

        timestamp = time.time()
        record = {
            "step": step,
            "timestamp": timestamp,
            **{f"{prefix}{k}": v for k, v in metrics.items()}
        }

        # Ghi vào metrics.jsonl (nếu bật)
        if self.save_metrics_jsonl:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        # Ghi vào TensorBoard
        if self._require_tb_writer("log_metrics"):
            for k, v in metrics.items():
                tag = f"{prefix}{k}" if prefix else k
                self.tb_writer.add_scalar(tag, v, global_step=step)

        # Ghi vào WandB
        if self.use_wandb:
            try:
                import wandb
                wandb_payload = {f"{prefix}{k}": v for k, v in metrics.items()}
                if "epoch" in metrics:
                    wandb_payload["epoch"] = metrics["epoch"]
                if prefix == "train/" and "train/step" not in wandb_payload:
                    wandb_payload["train/step"] = step
                wandb.log(wandb_payload, step=step)
            except Exception as e:
                self.error(f"[LỖI] Không thể ghi metrics lên WandB: {e}")
                raise

    def log_histogram(self, tag: str, values, step: int):
        """Ghi histogram trọng số hoặc gradient vào TensorBoard."""
        if not self._require_tb_writer("log_histogram"):
            return
        if not tag:
            self.error("[LỖI] log_histogram(): tag không được rỗng.")
            raise ValueError("log_histogram(): tag không được rỗng.")
        if values is None:
            self.error(f"[LỖI] log_histogram('{tag}'): values=None, không thể ghi histogram.")
            raise ValueError(f"log_histogram('{tag}'): values không được là None.")
        if isinstance(values, torch.Tensor):
            if values.numel() == 0:
                self.error(f"[LỖI] log_histogram('{tag}'): Tensor rỗng (numel=0).")
                raise ValueError(f"log_histogram('{tag}'): Tensor rỗng.")
            self.tb_writer.add_histogram(tag, values.detach().cpu(), global_step=step)
        else:
            arr = np.array(values)
            if arr.size == 0:
                self.error(f"[LỖI] log_histogram('{tag}'): Array rỗng (size=0).")
                raise ValueError(f"log_histogram('{tag}'): Array rỗng.")
            self.tb_writer.add_histogram(tag, arr, global_step=step)

    def log_image(self, tag: str, img_tensor, step: int):
        """
        Ghi ảnh vào TensorBoard và WandB.
        img_tensor: [C, H, W] hoặc [B, C, H, W] (lấy ảnh đầu tiên trong batch).
        """
        if not self.use_tensorboard and not self.use_wandb:
            return
        if not tag:
            self.error("[LỖI] log_image(): tag không được rỗng.")
            raise ValueError("log_image(): tag không được rỗng.")
        if img_tensor is None:
            self.error(f"[LỖI] log_image('{tag}'): img_tensor=None.")
            raise ValueError(f"log_image('{tag}'): img_tensor không được là None.")

        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)
        elif isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
        else:
            self.error(f"[LỖI] log_image('{tag}'): img_tensor có kiểu không hợp lệ {type(img_tensor)}.")
            raise ValueError(f"log_image('{tag}'): img_tensor phải là torch.Tensor hoặc np.ndarray.")

        if img_tensor.ndim not in (3, 4):
            self.error(f"[LỖI] log_image('{tag}'): img_tensor.ndim={img_tensor.ndim}, kỳ vọng 3 hoặc 4.")
            raise ValueError(f"log_image('{tag}'): ndim phải là 3 hoặc 4, nhận được {img_tensor.ndim}.")

        if img_tensor.ndim == 4:
            if img_tensor.shape[0] == 0:
                self.error(f"[LỖI] log_image('{tag}'): Batch rỗng.")
                raise ValueError(f"log_image('{tag}'): Batch rỗng.")
            img_tensor = img_tensor[0]  # Lấy ảnh đầu tiên trong batch

        # Tách RGB (3 kênh) và Depth (1 kênh) nếu là ảnh 4 kênh RGB-D
        if img_tensor.shape[0] == 4:
            rgb = img_tensor[:3]  # [3, H, W]
            depth = img_tensor[3:4].repeat(3, 1, 1)  # Lặp thành 3 kênh grayscale
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_image(f"{tag}/RGB", rgb, global_step=step)
                self.tb_writer.add_image(f"{tag}/Depth", depth, global_step=step)
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        f"{tag}/RGB": wandb.Image(rgb),
                        f"{tag}/Depth": wandb.Image(depth),
                    }, step=step)
                except Exception as e:
                    self.warning(f"[!] Không thể đẩy ảnh '{tag}' lên WandB: {e}")
        elif img_tensor.shape[0] in (1, 3):
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_image(tag, img_tensor, global_step=step)
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({tag: wandb.Image(img_tensor)}, step=step)
                except Exception as e:
                    self.warning(f"[!] Không thể đẩy ảnh '{tag}' lên WandB: {e}")
        else:
            self.error(f"[LỖI] log_image('{tag}'): Số kênh={img_tensor.shape[0]}, chỉ hỗ trợ 1, 3, hoặc 4.")
            raise ValueError(f"log_image('{tag}'): Số kênh không hợp lệ: {img_tensor.shape[0]}")

    def log_figure(self, tag: str, figure, step: int, close: bool = True):
        """Ghi biểu đồ matplotlib Figure vào TensorBoard và WandB."""
        if not tag:
            self.error("[LỖI] log_figure(): tag không được rỗng.")
            raise ValueError("log_figure(): tag không được rỗng.")
        if figure is None:
            self.error(f"[LỖI] log_figure('{tag}'): figure=None.")
            raise ValueError(f"log_figure('{tag}'): figure không được là None.")

        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_figure(tag, figure, global_step=step, close=False)

        if self.use_wandb:
            try:
                import wandb
                wandb.log({tag: wandb.Image(figure)}, step=step)
            except Exception as e:
                self.warning(f"[!] Không thể đẩy figure '{tag}' lên WandB: {e}")

        if close:
            import matplotlib.pyplot as plt
            plt.close(figure)

    def log_graph(self, model: nn.Module, input_data: tuple):
        """
        Ghi đồ thị kiến trúc mạng (Model Graph) vào TensorBoard.
        Nếu thất bại → raise để dừng ngay, KHÔNG bỏ qua.
        """
        if not self._require_tb_writer("log_graph"):
            return
        if model is None:
            self.error("[LỖI] log_graph(): model=None.")
            raise ValueError("log_graph(): model không được là None.")
        if input_data is None or len(input_data) == 0:
            self.error("[LỖI] log_graph(): input_data rỗng hoặc None.")
            raise ValueError("log_graph(): input_data không được rỗng.")
        self.tb_writer.add_graph(model, input_data)
        self.info("[✓] Đã ghi Model Graph vào TensorBoard.")

    def log_embedding(self, tag: str, mat, metadata: Optional[list] = None, step: int = 0):
        """Ghi embedding vectors vào TensorBoard Projector."""
        if not self._require_tb_writer("log_embedding"):
            return
        if not tag:
            self.error("[LỖI] log_embedding(): tag không được rỗng.")
            raise ValueError("log_embedding(): tag không được rỗng.")
        if mat is None:
            self.error(f"[LỖI] log_embedding('{tag}'): mat=None.")
            raise ValueError(f"log_embedding('{tag}'): mat không được là None.")

        if isinstance(mat, np.ndarray):
            mat = torch.from_numpy(mat)
        elif isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu()
        else:
            self.error(f"[LỖI] log_embedding('{tag}'): mat có kiểu không hợp lệ {type(mat)}.")
            raise ValueError(f"log_embedding('{tag}'): mat phải là torch.Tensor hoặc np.ndarray.")

        if mat.ndim != 2:
            self.error(f"[LỖI] log_embedding('{tag}'): mat.ndim={mat.ndim}, kỳ vọng 2.")
            raise ValueError(f"log_embedding('{tag}'): mat phải có 2 chiều [N, D], nhận được ndim={mat.ndim}.")
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            self.error(f"[LỖI] log_embedding('{tag}'): mat rỗng (shape={mat.shape}).")
            raise ValueError(f"log_embedding('{tag}'): mat rỗng.")
        if metadata is not None and len(metadata) != mat.shape[0]:
            self.error(f"[LỖI] log_embedding('{tag}'): len(metadata)={len(metadata)} != mat.shape[0]={mat.shape[0]}.")
            raise ValueError(f"log_embedding('{tag}'): metadata và mat phải cùng kích thước N.")
        self.tb_writer.add_embedding(mat, metadata=metadata, tag=tag, global_step=step)

    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Ghi hyperparameters và metrics cuối cùng vào HParams Dashboard."""
        if not self._require_tb_writer("log_hparams"):
            return
        if not hparam_dict:
            self.error("[LỖI] log_hparams(): hparam_dict rỗng.")
            raise ValueError("log_hparams(): hparam_dict không được rỗng.")
        if not metric_dict:
            self.error("[LỖI] log_hparams(): metric_dict rỗng.")
            raise ValueError("log_hparams(): metric_dict không được rỗng.")

        # Lọc kiểu dữ liệu hỗ trợ, BÁO LỖI RÕ RÀNG cho từng key bị bỏ qua
        clean_hparams = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hparams[k] = v
            elif v is None:
                self.warning(f"[CẢNH BÁO] log_hparams(): key '{k}' có giá trị None → bỏ qua.")
            else:
                self.warning(
                    f"[CẢNH BÁO] log_hparams(): key '{k}' có kiểu {type(v).__name__} "
                    f"(giá trị: {v!r}) không được TensorBoard hỗ trợ → bỏ qua."
                )

        if not clean_hparams:
            self.error("[LỖI] log_hparams(): Sau khi lọc, không còn hparam nào hợp lệ!")
            raise ValueError("log_hparams(): Không có hparam hợp lệ để ghi.")

        clean_metrics = {k: float(v) for k, v in metric_dict.items()}
        self.tb_writer.add_hparams(clean_hparams, clean_metrics)

    def setup_custom_scalars_layout(self):
        """Thiết lập layout Custom Scalars gộp nhóm cho dễ đọc."""
        if not self._require_tb_writer("setup_custom_scalars_layout"):
            return
        layout = {
            "Loss Overview": {
                "Total Loss (Train vs Val)": [
                    "Multiline", ["train_epoch/loss", "val_epoch/loss"]
                ],
                "Reconstruction L1 (Train vs Val)": [
                    "Multiline", ["train_epoch/l1", "val_epoch/l1"]
                ],
                "KL Divergence": [
                    "Multiline", ["train_epoch/kl", "val_epoch/kl"]
                ],
            },
            "Learning Rate": {
                "LR Schedule": [
                    "Multiline", ["train_epoch/lr"]
                ],
            },
            "Validation Quality": {
                "Max Joint MAE": [
                    "Multiline", ["val_epoch/max_joint_mae"]
                ],
            },
        }
        self.tb_writer.add_custom_scalars(layout)
        self.info("[✓] Đã thiết lập Custom Scalars Layout trên TensorBoard.")

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        elapsed_sec: float = 0.0,
    ):
        """
        In dòng tóm tắt Epoch đẹp mắt ra console và log file.
        """
        line = (
            f"Epoch [{epoch:4d}/{total_epochs:4d}] "
            f"| Train Loss: {train_metrics.get('loss', 0.0):.4f} "
            f"(L1: {train_metrics.get('l1', 0.0):.4f}, KL: {train_metrics.get('kl', 0.0):.4f}) "
        )

        if val_metrics is not None:
            line += (
                f"| Val Loss: {val_metrics.get('loss', 0.0):.4f} "
                f"(L1: {val_metrics.get('l1', 0.0):.4f}) "
            )

        line += f"| LR: {lr:.2e} | Time: {elapsed_sec:.1f}s"
        self.info(line)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                self.error(f"[LỖI] Không thể đóng kết nối WandB: {e}")
                raise
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
