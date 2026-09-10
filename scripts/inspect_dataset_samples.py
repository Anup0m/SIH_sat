"""
SatQuery AI — Dataset Inspection & Health Verification (Phase 2)
Inspects downloaded remote sensing samples, verifies image dimensions,
bounding boxes, and prints exact question-answer structures.
"""

import json
from pathlib import Path
import pandas as pd
from PIL import Image

from satquery_core.data_engine.cleaner import RemoteSensingCleaner

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
BEN_SAMPLE = SAMPLES_DIR / "bigearthnet"
VRS_SAMPLE = SAMPLES_DIR / "vrsbench"


def inspect_bigearthnet():
    print("\n========================================================")
    print("1. INSPECTING BIGEARTHNET.TXT SAMPLE")
    print("========================================================")
    parquet_path = BEN_SAMPLE / "sample_500.parquet"
    if not parquet_path.exists():
        # Fallback to full file if sample not sliced yet
        parquet_path = BASE_DIR / "data" / "raw" / "bigearthnet" / "BigEarthNet.txt.parquet"

    if not parquet_path.exists():
        print("[!] BigEarthNet.txt not found yet. Please wait for download to finish.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"[*] Loaded sample rows: {len(df):,}")
    print(f"[*] Columns: {list(df.columns)}")
    
    # Task distribution
    if "type" in df.columns:
        print("\n[*] Task types in sample:")
        print(df["type"].value_counts().to_string())

    print("\n[*] Sample Record #1:")
    sample = df.iloc[0].to_dict()
    for k, v in sample.items():
        print(f"    {k}: {v}")


def inspect_vrsbench():
    print("\n========================================================")
    print("2. INSPECTING VRSBENCH SAMPLE")
    print("========================================================")
    ann_file = VRS_SAMPLE / "sample_annotations.json"
    images_dir = VRS_SAMPLE / "images"

    if not ann_file.exists():
        print("[!] VRSBench annotations not found yet.")
        return

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[*] Total downloaded samples: {len(data)}")
    
    # Check images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"[*] Total image files on disk: {len(image_files)}")

    if data:
        first = data[0]
        print(f"\n[*] Sample #1 (ID: {first.get('image_id')}):")
        print(f"    Image: {first.get('image_file')}")
        img_p = images_dir / first.get("image_file", "")
        if img_p.exists():
            with Image.open(img_p) as im:
                print(f"    Resolution: {im.size} (Width x Height), Mode: {im.mode}")
        print(f"    Caption: {first.get('caption')[:120]}...")
        print(f"    Objects / Boxes: {first.get('objects')}")
        print(f"    VQA / Conversations: {first.get('vqa')}")


def run_cleaner_verification():
    print("\n========================================================")
    print("3. RUNNING DATA CLEANER & INTEGRITY CHECK")
    print("========================================================")
    cleaner = RemoteSensingCleaner()
    ann_file = VRS_SAMPLE / "sample_annotations.json"
    images_dir = VRS_SAMPLE / "images"

    if not ann_file.exists():
        print("[!] Cannot run cleaner: VRSBench data not ready yet.")
        return

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        img_path = images_dir / item.get("image_file", "")
        cleaner.clean_sample(image_path=img_path)

    report = cleaner.get_summary_report()
    print("[*] Cleaner Verification Results:")
    for k, v in report.items():
        print(f"    {k}: {v}")


def main():
    print("==========================================================")
    print("SatQuery AI — Dataset Inspection & Validation")
    print("==========================================================")
    inspect_bigearthnet()
    inspect_vrsbench()
    run_cleaner_verification()
    print("\n==========================================================")
    print("Inspection complete.")
    print("==========================================================")


if __name__ == "__main__":
    main()
