"""
SatQuery AI — Automated Data Cleaning & Sanitization Engine
Performs zero-compromise automated validation on satellite imagery and annotations:
1. File integrity checks (corrupted image files)
2. Signal validity (dead black tiles, sensor dropouts)
3. Cloud cover rejection (>80% saturated white pixels)
4. Bounding box validity (coordinates within image bounds, valid area)
5. Question & Answer string sanitization
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image


class RemoteSensingCleaner:
    def __init__(self, cloud_threshold: float = 0.85, min_std: float = 2.0):
        """
        Args:
            cloud_threshold: Fraction of pixels above 240/255 to flag as cloud-obscured.
            min_std: Minimum pixel standard deviation to ensure tile is not pure black or flat noise.
        """
        self.cloud_threshold = cloud_threshold
        self.min_std = min_std
        self.stats = {
            "total_inspected": 0,
            "valid_samples": 0,
            "rejected_corrupt": 0,
            "rejected_black_tile": 0,
            "rejected_cloud_cover": 0,
            "rejected_invalid_box": 0,
            "rejected_empty_text": 0
        }

    def validate_image_array(self, img_arr: np.ndarray) -> Tuple[bool, str]:
        """
        Checks if an image array is healthy for AI training.
        """
        if img_arr is None or img_arr.size == 0:
            return False, "Empty array"

        # Check for NaN or Inf
        if np.isnan(img_arr).any() or np.isinf(img_arr).any():
            return False, "Contains NaN or Inf values"

        # Check for dead / black sensor tiles
        std_val = float(np.std(img_arr))
        if std_val < self.min_std:
            return False, f"Dead/flat tile (std={std_val:.2f} < {self.min_std})"

        # Check for heavy cloud cover (optical RGB only)
        if img_arr.ndim >= 3 and img_arr.shape[-1] >= 3:
            # Over-saturated bright white pixels
            bright_pixels = np.all(img_arr[..., :3] > 240, axis=-1)
            cloud_ratio = float(np.mean(bright_pixels))
            if cloud_ratio > self.cloud_threshold:
                return False, f"Obscured by thick cloud cover ({cloud_ratio*100:.1f}%)"

        return True, "Valid"

    def validate_image_file(self, file_path: str | Path) -> Tuple[bool, str]:
        """Verifies if an image file exists, can be opened, and contains valid signal."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {path.name}"

        try:
            with Image.open(path) as img:
                img.verify()  # Verify PIL integrity
            # Reopen to read array (verify closes the file)
            with Image.open(path) as img:
                arr = np.array(img)
            return self.validate_image_array(arr)
        except Exception as e:
            return False, f"Corrupted image file: {str(e)}"

    def validate_bounding_box(
        self, 
        box: List[float], 
        img_width: int, 
        img_height: int,
        normalized: bool = False
    ) -> Tuple[bool, str]:
        """
        Validates a bounding box [ymin, xmin, ymax, xmax].
        Ensures non-negative coordinates, positive area, and within boundary.
        """
        if len(box) != 4:
            return False, f"Expected 4 coordinates, got {len(box)}"

        ymin, xmin, ymax, xmax = box

        max_h = 1.0 if normalized else img_height
        max_w = 1.0 if normalized else img_width

        if xmin < 0 or ymin < 0 or xmax > max_w or ymax > max_h:
            return False, f"Box [{ymin}, {xmin}, {ymax}, {xmax}] out of bounds (max: {max_h}x{max_w})"

        if xmax <= xmin or ymax <= ymin:
            return False, f"Degenerate box area: width={xmax - xmin}, height={ymax - ymin}"

        return True, "Valid"

    def validate_qa_pair(self, question: str, answer: str) -> Tuple[bool, str]:
        """Validates that question and answer are non-empty and well-formed."""
        if not question or not isinstance(question, str) or len(question.strip()) < 3:
            return False, "Question is empty or too short"

        if not answer or not isinstance(answer, str) or len(answer.strip()) < 1:
            return False, "Answer is empty or invalid"

        return True, "Valid"

    def clean_sample(
        self,
        image_path: str | Path,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        img_size: Tuple[int, int] = (512, 512)
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive check on an entire multi-modal sample.
        Returns: (is_valid, list_of_reasons_if_rejected)
        """
        self.stats["total_inspected"] += 1
        reasons = []

        # 1. Image check
        img_ok, img_reason = self.validate_image_file(image_path)
        if not img_ok:
            reasons.append(f"Image: {img_reason}")
            if "Corrupted" in img_reason:
                self.stats["rejected_corrupt"] += 1
            elif "Dead" in img_reason:
                self.stats["rejected_black_tile"] += 1
            elif "cloud" in img_reason.lower():
                self.stats["rejected_cloud_cover"] += 1

        # 2. QA check if provided
        if question is not None and answer is not None:
            qa_ok, qa_reason = self.validate_qa_pair(question, answer)
            if not qa_ok:
                reasons.append(f"QA: {qa_reason}")
                self.stats["rejected_empty_text"] += 1

        # 3. Box check if provided
        if boxes is not None:
            w, h = img_size
            for box in boxes:
                box_ok, box_reason = self.validate_bounding_box(box, w, h)
                if not box_ok:
                    reasons.append(f"Box: {box_reason}")
                    self.stats["rejected_invalid_box"] += 1
                    break

        if reasons:
            return False, reasons

        self.stats["valid_samples"] += 1
        return True, ["Valid"]

    def get_summary_report(self) -> Dict[str, Any]:
        """Returns cleaning statistics dictionary."""
        total = self.stats["total_inspected"]
        valid = self.stats["valid_samples"]
        pct = (valid / total * 100.0) if total > 0 else 0.0
        return {
            **self.stats,
            "pass_rate_percentage": round(pct, 2)
        }
