#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset-wide Resampling & Synchronization CLI Tool for OpenArm ACT
Processes raw HDF5 dataset containing variable dt / frame drops / jitter
and converts all episodes to strictly uniform 50.0 Hz datasets (control_dt = 0.020s).

Usage:
    python -m act_pipeline.data.resample_dataset --input_dir data_set/data_set --output_dir data_set/data_set_50hz
    python -m act_pipeline.data.resample_dataset --input_dir dataset --output_dir dataset_50hz --hz 50.0
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Any

from act_pipeline.data.resample import resample_hdf5_file, extract_timestamps
import h5py


def resample_dataset(
    input_dir: str,
    output_dir: str,
    target_hz: float = 50.0,
    compression: str = "gzip",
    compression_opts: int = 4,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    """
    Quét toàn bộ thư mục input_dir, lấy mẫu lại tất cả file .hdf5 sang output_dir.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    files = [f for f in files if not f.endswith(".partial.hdf5")]

    if len(files) == 0:
        print(f"[X] LỖI: Không tìm thấy file .hdf5 nào trong: {input_dir}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 85)
    print("   🔄 TIỆN ÍCH LẤY MẪU LẠI & ĐỒNG BỘ TẦN SỐ DATASET (TEMPORAL RESAMPLER @ 50Hz)")
    print("=" * 85)
    print(f"[*] Thư mục nguồn (Input)  : {input_dir}")
    print(f"[*] Thư mục đích  (Output) : {output_dir}")
    print(f"[*] Tần số mục tiêu        : {target_hz} Hz (chu kỳ dt = {1000.0/target_hz:.1f}ms)")
    print(f"[*] Tổng số episodes       : {len(files)}")
    print("=" * 85)

    reports = []
    total_orig_steps = 0
    total_new_steps = 0
    jitter_count = 0
    start_time = time.time()

    for idx, src_file in enumerate(files, 1):
        filename = os.path.basename(src_file)
        dst_file = os.path.join(output_dir, filename)

        if os.path.exists(dst_file) and not overwrite:
            print(f"[{idx:3d}/{len(files)}] {filename:<32} ⏩ ĐÃ TỒN TẠI (Bỏ qua, dùng --overwrite để ghi đè)")
            continue

        try:
            report = resample_hdf5_file(
                src_path=src_file,
                dst_path=dst_file,
                target_hz=target_hz,
                compression=compression,
                compression_opts=compression_opts,
            )
            reports.append(report)
            total_orig_steps += report["original_samples"]
            total_new_steps += report["resampled_samples"]

            status = "⚠️ ĐÃ KHẮC PHỤC RỚT KHUNG" if report["jitter_detected"] else "✅ ĐẠT CHUẨN"
            if report["jitter_detected"]:
                jitter_count += 1

            print(
                f"[{idx:3d}/{len(files)}] {filename:<30} {status} | "
                f"{report['original_samples']} -> {report['resampled_samples']} steps "
                f"({report['actual_hz']:.1f}Hz -> {target_hz:.1f}Hz | Rớt: {report['drop_pct']:.1f}%)"
            )
        except Exception as e:
            print(f"[{idx:3d}/{len(files)}] {filename:<30} ❌ LỖI: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"🎉 HOÀN TẤT ĐỒNG BỘ HÓA DATASET TRONG {elapsed:.1f} GIÂY!")
    print(f"   + Tổng số episodes xử lý : {len(reports)}/{len(files)}")
    print(f"   + Episodes có jitter/rớt : {jitter_count} episodes (đã nội suy bù đủ 50Hz)")
    print(f"   + Tổng timesteps         : {total_orig_steps} -> {total_new_steps} steps")
    print(f"   + Thư mục dataset chuẩn  : {output_dir}")
    print("=" * 85 + "\n")

    return reports


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lấy mẫu lại toàn bộ dataset HDF5 về tần số đồng đều chuẩn 50.0 Hz (khắc phục Frame Drop / Jitter)"
    )
    parser.add_argument("--input_dir", type=str, default="data_set/data_set", help="Thư mục dataset gốc bị rớt khung hình")
    parser.add_argument("--output_dir", type=str, default="data_set/data_set_50hz", help="Thư mục lưu dataset sau khi lấy mẫu lại")
    parser.add_argument("--hz", type=float, default=50.0, help="Tần số lấy mẫu mong muốn (mặc định: 50.0 Hz)")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè file nếu đã tồn tại")
    return parser.parse_args()


def main():
    args = parse_args()
    resample_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_hz=args.hz,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

