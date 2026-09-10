"""
SatQuery AI — Sample Dataset Downloader (Phase 2)
Downloads lightweight local developer samples (<1.5 GB total) for local testing and inspection.
Zero-compromise engineering: fully reproducible, robust error handling, progress bars.
"""

import os
import sys
import json
import time
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"

BEN_RAW = RAW_DIR / "bigearthnet"
BEN_SAMPLE = SAMPLES_DIR / "bigearthnet"
VRS_SAMPLE = SAMPLES_DIR / "vrsbench"
RSVQA_SAMPLE = SAMPLES_DIR / "rsvqa"
CDVQA_SAMPLE = SAMPLES_DIR / "cdvqa"


def ensure_dirs():
    """Ensure all required directories exist."""
    for p in [RAW_DIR, SAMPLES_DIR, BEN_RAW, BEN_SAMPLE, VRS_SAMPLE, RSVQA_SAMPLE, CDVQA_SAMPLE]:
        p.mkdir(parents=True, exist_ok=True)


def download_file_with_progress(url: str, dest_path: Path, description: str = "Downloading"):
    """Downloads a file with a live progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[*] Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return dest_path

    print(f"\n[+] Starting: {description}")
    print(f"    URL: {url}")
    print(f"    Saving to: {dest_path}")

    headers = {"User-Agent": "SatQuery-AI-Downloader/1.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB chunk

    progress = tqdm(
        total=total_size, 
        unit="iB", 
        unit_scale=True, 
        desc=dest_path.name,
        ncols=80
    )

    temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with open(temp_dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()

    temp_dest.rename(dest_path)
    print(f"[✓] Download completed: {dest_path.name}")
    return dest_path


def download_bigearthnet_metadata():
    """
    Downloads the BigEarthNet.txt instruction-tuning dataset (~467 MB).
    Then extracts a 500-sample slice for quick local development.
    """
    print("\n========================================================")
    print("STEP 1: BigEarthNet.txt (Primary RS Adaptation Dataset)")
    print("========================================================")
    
    url = "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet"
    dest_file = BEN_RAW / "BigEarthNet.txt.parquet"
    
    try:
        download_file_with_progress(url, dest_file, "BigEarthNet.txt Parquet Annotations")
        
        # Slicing a fast 500-sample test set for local dev
        sample_path = BEN_SAMPLE / "sample_500.parquet"
        if not sample_path.exists():
            print(f"[*] Extracting 500-row sample for rapid local dev...")
            df = pd.read_parquet(dest_file)
            print(f"    Total rows in full file: {len(df):,}")
            print(f"    Available columns: {list(df.columns)}")
            
            sample_df = df.head(500)
            sample_df.to_parquet(sample_path)
            
            # Also save a readable JSON of first 5 samples for inspection
            sample_json = BEN_SAMPLE / "sample_first_5.json"
            sample_df.head(5).to_json(sample_json, orient="records", indent=2)
            print(f"[✓] Saved local sample to: {sample_path}")
            print(f"[✓] Saved inspection JSON to: {sample_json}")
        else:
            print(f"[*] Local sample already ready: {sample_path}")
            
    except Exception as e:
        print(f"[!] Error processing BigEarthNet.txt: {e}")


def download_vrsbench_sample(num_samples: int = 50):
    """
    Streams the first num_samples from xiang709/VRSBench via Hugging Face.
    Saves images (512x512) and groundings/captions/VQA pairs.
    """
    print("\n========================================================")
    print("STEP 2: VRSBench (Single-Image VQA, Grounding, Captioning)")
    print("========================================================")
    
    images_dir = VRS_SAMPLE / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_file = VRS_SAMPLE / "sample_annotations.json"
    
    if ann_file.exists() and len(list(images_dir.glob("*.jpg"))) >= num_samples:
        print(f"[*] VRSBench sample already downloaded ({num_samples} samples).")
        return

    print(f"[*] Streaming {num_samples} samples from xiang709/VRSBench...")
    try:
        from datasets import load_dataset
        ds = load_dataset("xiang709/VRSBench", split="train", streaming=True)
        
        records = []
        count = 0
        pbar = tqdm(total=num_samples, desc="VRSBench Samples", ncols=80)
        
        for item in ds:
            img = item.get("image")
            img_id = item.get("image_id", f"vrs_{count:04d}")
            img_filename = f"{img_id}.jpg"
            img_path = images_dir / img_filename
            
            if img is not None:
                img.save(img_path, format="JPEG")
            
            # Extract metadata without the raw PIL image object
            record = {
                "image_id": img_id,
                "image_file": img_filename,
                "caption": item.get("caption", ""),
                "objects": item.get("objects", []),
                "vqa": item.get("vqa", item.get("conversations", []))
            }
            records.append(record)
            
            count += 1
            pbar.update(1)
            if count >= num_samples:
                break
                
        pbar.close()
        
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            
        print(f"[✓] Saved {count} VRSBench images to: {images_dir}")
        print(f"[✓] Saved VRSBench annotations to: {ann_file}")
        
    except Exception as e:
        print(f"[!] Error streaming VRSBench: {e}")


def main():
    print("==========================================================")
    print("SatQuery AI — Local Sample Data Pipeline (Phase 2)")
    print("==========================================================")
    ensure_dirs()
    
    # 1. Download BigEarthNet.txt metadata
    download_bigearthnet_metadata()
    
    # 2. Download VRSBench samples
    download_vrsbench_sample(num_samples=50)
    
    print("\n==========================================================")
    print("Sample downloads finished! Next step: data inspection & cleaning.")
    print("==========================================================")


if __name__ == "__main__":
    main()
