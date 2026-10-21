#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Checkpoint Management for ACT
Handles:
1. Saving best_checkpoint.pth (based on lowest validation loss / L1 loss)
2. Saving latest_checkpoint.pth (continuous recovery point)
3. Periodic checkpoints: checkpoint_epoch_{N}.pth
4. Saving clean deployment weights: act_deploy_weights.pth
5. Resuming training seamlessly (restores model, optimizer, lr scheduler, scaler, stats, epoch)
"""

import os
import glob
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn


from act_pipeline.utils.onnx_export import export_act_to_onnx


class CheckpointManager:
    """
    Quản lý lưu trữ và khôi phục checkpoint mô hình ACT.
    Hỗ trợ cả định dạng PyTorch (.pth) phục vụ huấn luyện và ONNX (.onnx) phục vụ triển khai thực tế.
    """
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
        self.latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        self.deploy_weights_path = os.path.join(self.checkpoint_dir, "act_deploy_weights.pth")

        # Đường dẫn xuất file ONNX tương ứng
        self.best_onnx_path = os.path.join(self.checkpoint_dir, "best_checkpoint.onnx")
        self.latest_onnx_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.onnx")
        self.deploy_onnx_path = os.path.join(self.checkpoint_dir, "act_deploy_model.onnx")

    def export_to_onnx(
        self,
        model: nn.Module,
        output_path: Optional[str] = None,
        config: Optional[Any] = None,
        stats: Optional[Dict[str, Any]] = None,
        opset_version: int = 18,
        dynamic_batch: bool = True,
        verify_with_ort: bool = True,
    ) -> str:
        """
        Xuất mô hình ACT sang định dạng ONNX.
        """
        target_path = output_path if output_path is not None else self.best_onnx_path
        return export_act_to_onnx(
            model=model,
            output_path=target_path,
            config=config,
            stats=stats,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verify_with_ort=verify_with_ort,
        )

    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: Optional[Any],
        stats: Dict[str, Any],
        config: Dict[str, Any],
        metrics: Dict[str, float],
        is_best: bool = False,
        save_periodic: bool = False,
        save_onnx: bool = True,
        save_pth: bool = False,
        save_periodic_onnx: bool = False,
        opset_version: int = 18,
    ) -> Dict[str, str]:
        """
        Lưu checkpoint mô hình ACT.
        Theo cấu hình mặc định tinh gọn, chỉ lưu duy nhất best_checkpoint.onnx
        và best_checkpoint_metadata.json (cùng dataset_stats.pkl) để triển khai trực tiếp.
        """
        saved_paths = {}

        # 1. Chỉ lưu các file PyTorch (.pth) nếu save_pth=True
        if save_pth:
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            checkpoint_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "stats": stats,
                "config": config,
                "metrics": metrics,
            }
            torch.save(checkpoint_data, self.latest_checkpoint_path)
            saved_paths["latest"] = self.latest_checkpoint_path

            if is_best:
                torch.save(checkpoint_data, self.best_checkpoint_path)
                saved_paths["best"] = self.best_checkpoint_path

            if save_periodic:
                periodic_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth")
                torch.save(checkpoint_data, periodic_path)
                saved_paths["periodic"] = periodic_path

        # 2. Xuất mô hình ONNX khi đạt Best Checkpoint
        if is_best and save_onnx:
            try:
                export_act_to_onnx(
                    model=model,
                    output_path=self.best_onnx_path,
                    config=config,
                    stats=stats,
                    opset_version=opset_version,
                    dynamic_batch=True,
                    verify_with_ort=True,
                )
                saved_paths["best_onnx"] = self.best_onnx_path
                saved_paths["metadata"] = os.path.splitext(self.best_onnx_path)[0] + "_metadata.json"
            except Exception as exc:
                print(f"[!] Cảnh báo: Xuất ONNX cho Best Checkpoint thất bại: {exc}")

        # 3. Xuất ONNX định kỳ nếu yêu cầu
        if save_periodic and save_periodic_onnx and save_onnx:
            periodic_onnx_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.onnx")
            try:
                export_act_to_onnx(
                    model=model,
                    output_path=periodic_onnx_path,
                    config=config,
                    stats=stats,
                    opset_version=opset_version,
                    dynamic_batch=True,
                )
                saved_paths["periodic_onnx"] = periodic_onnx_path
            except Exception as exc:
                print(f"[!] Cảnh báo: Xuất ONNX định kỳ epoch {epoch} thất bại: {exc}")

        return saved_paths

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Khôi phục toàn bộ trạng thái từ một file checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {checkpoint_path}")

        print(f"[*] Đang tải checkpoint từ: {checkpoint_path} lên {device}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 1. Tải trọng số mô hình
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Nếu model bọc trong DataParallel
        target_model = model.module if hasattr(model, "module") else model
        missing, unexpected = target_model.load_state_dict(state_dict, strict=strict)
        if len(missing) > 0:
            print(f"[!] Cảnh báo các keys bị thiếu khi load: {missing}")
        if len(unexpected) > 0:
            print(f"[!] Cảnh báo các keys thừa: {unexpected}")

        # 2. Khôi phục Optimizer nếu được cung cấp
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[✓] Đã khôi phục trạng thái Optimizer.")

        # 3. Khôi phục Scheduler nếu được cung cấp
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("[✓] Đã khôi phục trạng thái Learning Rate Scheduler.")

        # 4. Khôi phục GradScaler nếu có
        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("[✓] Đã khôi phục trạng thái GradScaler (AMP).")

        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        stats = checkpoint.get("stats", None)
        config = checkpoint.get("config", {})
        metrics = checkpoint.get("metrics", {})

        print(f"[✓] Đã khôi phục thành công checkpoint tại Epoch {epoch}, Global Step {global_step}!")
        return {
            "epoch": epoch,
            "global_step": global_step,
            "stats": stats,
            "config": config,
            "metrics": metrics,
        }

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Tự động tìm checkpoint mới nhất trong thư mục để khôi phục (Resume).
        """
        if os.path.exists(self.latest_checkpoint_path):
            return self.latest_checkpoint_path

        periodic_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")))
        if len(periodic_files) > 0:
            return periodic_files[-1]

        if os.path.exists(self.best_checkpoint_path):
            return self.best_checkpoint_path

        return None
