"""
SatQuery AI — Unified Training & Evaluation Launcher (Phase 4 & 9)
One-command entry point for all model training and evaluation tasks.
Automatically inspects available GPU hardware, applies memory optimizations,
and coordinates resilient training runs.
"""

import sys
import argparse
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.train_vlm import train as train_vlm
from training.train_change import train_change_model
from training.evaluate_benchmarks import run_all_evaluations


def inspect_hardware():
    print("\n========================================================")
    print("SATQUERY AI — HARDWARE DIAGNOSTICS")
    print("========================================================")
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available: {cuda_available}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] Primary GPU: {device_name}")
        print(f"[*] Total VRAM: {total_vram:.2f} GB")
        if total_vram <= 6.5:
            print("[!] Detected 6GB GPU (e.g. RTX 4050). Batch size 2 and LoRA recommended.")
        else:
            print("[+] High-capacity GPU detected. Standard training batches enabled.")
    else:
        print("[!] No CUDA GPU detected. Running on CPU mode.")
    print("========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Master Launcher")
    parser.add_argument(
        "--task", 
        choices=["vlm", "change", "eval", "all"], 
        default="eval",
        help="Task to execute: 'vlm' (RS adaptation), 'change' (Siamese detector), 'eval' (benchmarks), 'all'"
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity run on tiny sample")
    parser.add_argument("--resume", action="store_true", help="Resume from latest saved checkpoint")
    args = parser.parse_args()

    inspect_hardware()

    if args.task in ["vlm", "all"]:
        print("\n>>> Launching RS-VLM Adaptation Pipeline...")
        train_vlm(config_path="configs/training_vlm.yaml", dry_run=args.dry_run, resume=args.resume)

    if args.task in ["change", "all"]:
        print("\n>>> Launching Bi-Temporal Change Detection Pipeline...")
        train_change_model(config_path="configs/training_change.yaml", dry_run=args.dry_run, resume=args.resume)

    if args.task in ["eval", "all"]:
        print("\n>>> Launching Benchmark Evaluation...")
        run_all_evaluations()


if __name__ == "__main__":
    main()
