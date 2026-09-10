"""
SatQuery AI — Input Validation Engine (Phase 5)
Validates uploaded imagery, file formats, spatial compatibility, and image count
before executing agent workflows.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image

from satquery_core.schemas import AnalysisRequest


class InputValidator:
    """Validates inputs against SIH remote sensing criteria."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def validate_request(self, request: AnalysisRequest) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Validates the incoming AnalysisRequest.
        Returns:
            (is_valid: bool, error_message: str, image_metadata: List[Dict])
        """
        # 1. Query check
        if not request.query or not request.query.strip():
            return False, "Query cannot be empty. Please ask a question or specify a task.", []

        # 2. Image count check (SIH supports 1 image or 2 image pairs)
        if not request.image_paths:
            return False, "No image files provided. Please upload at least 1 image.", []

        if len(request.image_paths) > 2:
            return False, f"Maximum 2 images supported (single image or bi-temporal/cross-modal pair). Got {len(request.image_paths)}.", []

        # 3. File existence and format checks
        image_metadata = []
        for idx, p_str in enumerate(request.image_paths):
            p = Path(p_str)
            if not p.exists():
                return False, f"Image file not found: {p.name}", []

            if p.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported image extension '{p.suffix}'. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}", []

            try:
                with Image.open(p) as img:
                    w, h = img.size
                    meta = {
                        "image_index": idx + 1,
                        "filename": p.name,
                        "width": w,
                        "height": h,
                        "format": img.format,
                        "mode": img.mode
                    }
                    image_metadata.append(meta)
            except Exception as e:
                return False, f"Cannot open image {p.name}: {str(e)}", []

        # 4. Dimension compatibility for paired images
        if len(image_metadata) == 2:
            m1, m2 = image_metadata[0], image_metadata[1]
            if (m1["width"], m1["height"]) != (m2["width"], m2["height"]):
                return False, (
                    f"Dimension mismatch between paired images: "
                    f"Image 1 is ({m1['width']}x{m1['height']}), Image 2 is ({m2['width']}x{m2['height']}). "
                    f"Spatial pairs must have matching spatial extents."
                ), image_metadata

        return True, "Input validation passed successfully.", image_metadata
