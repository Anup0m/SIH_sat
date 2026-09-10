"""
SatQuery AI — Realistic Mock Specialists
These mock tools allow the entire software system (Agent, API, Frontend, Reports, Trace)
to be fully constructed, integrated, and verified offline without heavy GPU hardware.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw

from satquery_core.schemas import AnalysisRequest, VisualEvidence
from satquery_core.models.base_specialist import BaseSpecialist, SpecialistOutput


class MockVLMSpecialist(BaseSpecialist):
    """Mock Vision-Language Model for VQA, Scene Captioning, and Cross-Modal Queries."""

    def __init__(self):
        super().__init__(name="RS-VLM-Adapted", version="Domain-Adapted-LoRA", is_mock=True)

    def supports_task(self, task: str) -> bool:
        return task in ["vqa", "captioning", "cross_modal_analysis"]

    def supports_modality(self, modalities: List[str]) -> bool:
        return True

    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        query_lower = request.query.lower()

        # Context-aware realistic remote sensing responses
        if "changed" in query_lower and len(request.image_paths) == 1:
            answer = (
                "Change Detection Advisory: A bi-temporal image pair (T1 Before and T2 After) is required to compute "
                "surface change heatmaps and difference metrics. Only 1 image was uploaded. "
                "Please upload a second observation image or use the '⚡ Bi-Temporal Change' demo preset."
            )
            confidence = 0.88
        elif "water" in query_lower or "lake" in query_lower or "river" in query_lower:
            answer = "Water body identified in scene. Spectral reflectance shows low NIR response and strong radar signal absorption, indicating clear water boundaries."
            confidence = 0.92
        elif "urban" in query_lower or "building" in query_lower or "built-up" in query_lower:
            answer = "High density of built-up residential structures and paved road networks identified across the central sector."
            confidence = 0.89
        elif "sar" in query_lower or "radar" in query_lower:
            answer = "Joint Optical + SAR Analysis: Optical imagery provides true-color surface classification, while SAR backscatter reveals structural roughness and moisture boundaries under cloud penetration."
            confidence = 0.94
        elif task == "captioning" or "describe" in query_lower:
            answer = "Satellite scene displays mixed land cover: central park reservoir surrounded by high-density urban infrastructure, commercial high-rises, and perimeter roadway corridors."
            confidence = 0.91
        else:
            answer = f"Remote sensing analysis for query '{request.query}': The scene exhibits balanced terrestrial features with characteristic vegetative and urban signatures."
            confidence = 0.86

        return SpecialistOutput(
            answer=answer,
            evidence=VisualEvidence(evidence_type="none"),
            confidence_score=confidence,
            metadata={"specialist": self.name, "task": task, "status": "evaluation"}
        )


class MockGroundingSpecialist(BaseSpecialist):
    """Mock Spatial Grounding Specialist for object localization and bounding boxes."""

    def __init__(self):
        super().__init__(name="RS-Grounding-Head", version="Spatial-Grounding-v1.0", is_mock=True)

    def supports_task(self, task: str) -> bool:
        return task in ["grounding", "localization"]

    def supports_modality(self, modalities: List[str]) -> bool:
        return "optical" in modalities or len(modalities) == 0

    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        query_lower = request.query.lower()
        
        # Grounding coordinates normalized [ymin, xmin, ymax, xmax]
        if "water" in query_lower or "lake" in query_lower or "river" in query_lower:
            boxes = [
                [0.42, 0.44, 0.62, 0.54]
            ]
            labels = ["Water Body (Central Reservoir)"]
            answer = "Located central water reservoir at [ymin: 0.42, xmin: 0.44, ymax: 0.62, xmax: 0.54] with high optical absorption confidence."
        elif "building" in query_lower or "urban" in query_lower or "structure" in query_lower:
            boxes = [
                [0.10, 0.10, 0.35, 0.40],
                [0.12, 0.60, 0.38, 0.90],
                [0.65, 0.65, 0.90, 0.92]
            ]
            labels = ["Urban Cluster North-West", "Commercial Sector East", "Residential Complex South-East"]
            answer = "Located 3 prominent structural clusters across the northwestern, eastern, and southeastern quadrants."
        else:
            boxes = [[0.35, 0.35, 0.65, 0.65]]
            labels = [f"Target: {request.query[:25]}"]
            answer = f"Identified primary region of interest matching query: '{request.query}'."

        evidence = VisualEvidence(
            evidence_type="bounding_boxes",
            bounding_boxes=boxes,
            labels=labels,
            spatial_metadata={"coordinate_system": "Normalized (0.0 - 1.0)", "objects_detected": len(boxes)}
        )

        return SpecialistOutput(
            answer=answer,
            evidence=evidence,
            confidence_score=0.91,
            metadata={"specialist": self.name, "task": "grounding"}
        )


class MockChangeSpecialist(BaseSpecialist):
    """Mock Bi-Temporal Change Specialist: computes difference heatmaps between 2 images."""

    def __init__(self):
        super().__init__(name="RS-Change-Detector", version="Siamese-Difference-Engine", is_mock=True)

    def supports_task(self, task: str) -> bool:
        return task in ["change_detection", "change_description", "change_vqa"]

    def supports_modality(self, modalities: List[str]) -> bool:
        return "bitemporal_pair" in modalities or len(modalities) >= 2

    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        img_paths = request.image_paths
        mask_rel_path = None
        change_pct = 14.8

        if len(img_paths) >= 2 and Path(img_paths[0]).exists() and Path(img_paths[1]).exists():
            try:
                im1 = Image.open(img_paths[0]).convert("RGB").resize((512, 512))
                im2 = Image.open(img_paths[1]).convert("RGB").resize((512, 512))
                
                arr1 = np.array(im1, dtype=np.float32)
                arr2 = np.array(im2, dtype=np.float32)
                
                diff = np.mean(np.abs(arr1 - arr2), axis=-1)
                mask = (diff > 35.0).astype(np.uint8) * 255
                change_pct = round(float(np.mean(mask > 0) * 100.0), 2)

                out_dir = Path("data/samples/output_masks")
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Create glowing colored heatmap overlay onto im2
                im2_rgba = im2.convert("RGBA")
                overlay_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
                # Neon orange/red glow for change pixels
                overlay_rgba[mask > 0] = [255, 65, 0, 190]
                colored_overlay = Image.fromarray(overlay_rgba, mode="RGBA")
                blended = Image.alpha_composite(im2_rgba, colored_overlay)
                
                overlay_name = "change_overlay_latest.png"
                blended.save(out_dir / overlay_name)
                mask_rel_path = f"/static/output_masks/{overlay_name}"
            except Exception:
                pass

        answer = (
            f"Bi-temporal change analysis identified surface transition across {change_pct}% of the surveyed scene. "
            "Visual evidence indicates localized structural development and ground clearing between observation dates."
        )

        evidence = VisualEvidence(
            evidence_type="change_mask",
            mask_path=mask_rel_path,
            spatial_metadata={
                "change_percentage": change_pct,
                "timestamps": request.timestamps or ["Time 1", "Time 2"]
            }
        )

        return SpecialistOutput(
            answer=answer,
            evidence=evidence,
            confidence_score=0.90,
            metadata={"specialist": self.name, "task": "change_detection"}
        )
