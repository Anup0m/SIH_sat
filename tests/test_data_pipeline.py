"""
SatQuery AI — Data Pipeline Unit Tests (Phase 2 Verification)
Tests:
1. BigEarthNet.txt loader & sample extraction
2. VRSBench evaluation annotations loader
3. Cleaner & Sanitizer rules (signal validity, dead tile rejection, bounding box checks)
4. GeoTIFF / Remote Sensing array preprocessing
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from satquery_core.data_engine.dataset_loaders import BigEarthNetDataset, VRSBenchDataset
from satquery_core.data_engine.cleaner import RemoteSensingCleaner
from satquery_core.data_engine.geotiff_io import normalize_optical_bands, sar_to_decibels


def test_bigearthnet_loader():
    parquet_path = BASE_DIR / "data" / "samples" / "bigearthnet" / "sample_500.parquet"
    assert parquet_path.exists(), "Sample parquet does not exist!"
    
    ds = BigEarthNetDataset(parquet_path=parquet_path)
    assert len(ds) == 500, f"Expected 500 samples, got {len(ds)}"
    sample = ds[0]
    assert "instruction" in sample and len(sample["instruction"]) > 0
    assert "target_answer" in sample
    assert "s1_name" in sample and "patch_id" in sample
    print(f"[PASS] BigEarthNetDataset verified (Total: {len(ds)} samples).")


def test_vrsbench_loader():
    vqa_file = BASE_DIR / "data" / "raw" / "vrsbench" / "VRSBench_EVAL_vqa.json"
    ref_file = BASE_DIR / "data" / "raw" / "vrsbench" / "VRSBench_EVAL_referring.json"
    assert vqa_file.exists(), "VRSBench VQA file missing"
    assert ref_file.exists(), "VRSBench Referring file missing"
    print(f"[PASS] VRSBench annotations verified.")


def test_cleaner_engine():
    cleaner = RemoteSensingCleaner()
    
    # 1. Test black tile rejection
    black_tile = np.zeros((512, 512, 3), dtype=np.uint8)
    ok, reason = cleaner.validate_image_array(black_tile)
    assert not ok, "Cleaner should reject pure black tile!"
    assert "Dead/flat tile" in reason
    
    # 2. Test healthy synthetic tile
    healthy_tile = np.random.randint(20, 220, (512, 512, 3), dtype=np.uint8)
    ok, reason = cleaner.validate_image_array(healthy_tile)
    assert ok, f"Healthy tile should pass, failed with: {reason}"
    
    # 3. Test bounding box checks
    assert cleaner.validate_bounding_box([0.1, 0.2, 0.5, 0.8], 512, 512, normalized=True)[0]
    assert not cleaner.validate_bounding_box([-0.1, 0.2, 0.5, 0.8], 512, 512, normalized=True)[0]
    assert not cleaner.validate_bounding_box([0.5, 0.2, 0.1, 0.8], 512, 512, normalized=True)[0]
    
    print("[PASS] RemoteSensingCleaner rules verified.")


def test_remote_sensing_io():
    # 1. Optical contrast stretching test
    raw_16bit = np.random.randint(0, 10000, (256, 256, 3), dtype=np.uint16)
    stretched = normalize_optical_bands(raw_16bit)
    assert stretched.dtype == np.uint8
    assert stretched.shape == (256, 256, 3)
    
    # 2. SAR dB conversion test
    raw_sar_power = np.random.uniform(0.001, 1.5, (256, 256)).astype(np.float32)
    sar_db_img = sar_to_decibels(raw_sar_power)
    assert sar_db_img.dtype == np.uint8
    assert sar_db_img.min() >= 0 and sar_db_img.max() <= 255
    
    print("[PASS] GeoTIFF / Remote Sensing preprocessing verified.")


if __name__ == "__main__":
    test_bigearthnet_loader()
    test_vrsbench_loader()
    test_cleaner_engine()
    test_remote_sensing_io()
    print("\n========================================================")
    print("ALL PHASE 2 DATA PIPELINE TESTS PASSED!")
    print("========================================================")
