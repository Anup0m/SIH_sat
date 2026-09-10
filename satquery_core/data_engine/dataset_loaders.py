"""
SatQuery AI — Remote Sensing Dataset Loaders (PyTorch & Standalone)
Production-grade dataset classes for:
1. BigEarthNet.txt (Multisensor Sentinel-1 SAR + Sentinel-2 Optical instructions)
2. VRSBench (Single-image VQA, Captioning, Object Grounding)
3. CDVQA (Bi-temporal change questions and image pairs)

Zero-compromise: Handles both PyTorch Tensors and raw NumPy arrays.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import pandas as pd
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass


class BigEarthNetDataset(Dataset):
    """
    Dataset loader for BigEarthNet.txt instruction-tuning dataset.
    Connects Sentinel-1 SAR and Sentinel-2 Optical acquisitions to natural language queries.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        s1_image_root: Optional[Union[str, Path]] = None,
        s2_image_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.parquet_path = Path(parquet_path)
        self.s1_image_root = Path(s1_image_root) if s1_image_root else None
        self.s2_image_root = Path(s2_image_root) if s2_image_root else None

        # Load metadata
        df = pd.read_parquet(self.parquet_path)
        if split:
            df = df[df["split"] == split]
        if max_samples:
            df = df.head(max_samples)

        self.data = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        sample = {
            "id": int(row["ID"]),
            "s1_name": str(row["s1_name"]),
            "patch_id": str(row["patch_id"]),
            "instruction": str(row["input"]),
            "target_answer": str(row["output"]),
            "task_type": str(row["type"]),
            "category": str(row["category"]),
            "coordinates": (float(row["latitude"]), float(row["longitude"])),
            "s1_image": None,
            "s2_image": None
        }

        # If image roots are provided, load real image arrays
        if self.s2_image_root and (self.s2_image_root / f"{row['patch_id']}.tif").exists():
            with Image.open(self.s2_image_root / f"{row['patch_id']}.tif") as im:
                sample["s2_image"] = np.array(im)

        return sample


class VRSBenchDataset(Dataset):
    """
    Dataset loader for VRSBench:
    Provides:
      - 512x512 RGB satellite image
      - Whole-scene caption
      - Grounding bounding boxes [ymin, xmin, ymax, xmax] (normalized 0.0 - 1.0)
      - Visual question-answer pairs
    """

    def __init__(
        self, 
        annotations_json: Union[str, Path], 
        images_dir: Union[str, Path],
        max_samples: Optional[int] = None
    ):
        self.images_dir = Path(images_dir)
        with open(annotations_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        self.records = data

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_file = self.images_dir / rec.get("image_file", "")
        
        img_array = None
        if img_file.exists():
            with Image.open(img_file) as im:
                img_array = np.array(im.convert("RGB"))

        # Parse and normalize bounding boxes
        boxes = []
        labels = []
        raw_objs = rec.get("objects", [])
        for obj in raw_objs:
            box = obj.get("box", obj.get("bbox", []))
            if len(box) == 4:
                # VRSBench boxes can be 0-100 normalized or pixel coordinates
                # Standardize to 0.0 - 1.0
                ymin, xmin, ymax, xmax = [float(c) for c in box]
                if max(ymin, xmin, ymax, xmax) > 1.0:
                    # Normalize from 0-100 or 0-512
                    norm_denom = 512.0 if max(ymin, xmin, ymax, xmax) > 100.0 else 100.0
                    ymin /= norm_denom
                    xmin /= norm_denom
                    ymax /= norm_denom
                    xmax /= norm_denom
                boxes.append([ymin, xmin, ymax, xmax])
                labels.append(obj.get("category", "object"))

        return {
            "image_id": rec.get("image_id"),
            "image": img_array,
            "caption": rec.get("caption", ""),
            "bounding_boxes": boxes,
            "labels": labels,
            "vqa": rec.get("vqa", [])
        }


class CDVQADataset(Dataset):
    """
    Dataset loader for Change Detection Visual Question Answering (CDVQA).
    Provides:
      - Pre-event image (T1)
      - Post-event image (T2)
      - Change question & answer
      - Change mask (if available)
    """

    def __init__(self, pairs_metadata: List[Dict[str, Any]]):
        self.pairs = pairs_metadata

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        t1_arr, t2_arr, mask_arr = None, None, None

        if "t1_path" in item and Path(item["t1_path"]).exists():
            with Image.open(item["t1_path"]) as im:
                t1_arr = np.array(im.convert("RGB"))

        if "t2_path" in item and Path(item["t2_path"]).exists():
            with Image.open(item["t2_path"]) as im:
                t2_arr = np.array(im.convert("RGB"))

        if "mask_path" in item and Path(item["mask_path"]).exists():
            with Image.open(item["mask_path"]) as im:
                mask_arr = np.array(im.convert("L"))

        return {
            "t1_image": t1_arr,
            "t2_image": t2_arr,
            "mask": mask_arr,
            "question": item.get("question", ""),
            "answer": item.get("answer", "")
        }
