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
    """Vision-Language Model for dynamic VQA, Scene Captioning, and Cross-Modal Queries."""

    def __init__(self):
        super().__init__(name="RS-VLM-Adapted", version="Domain-Adapted-LoRA", is_mock=True)

    def supports_task(self, task: str) -> bool:
        return task in ["vqa", "captioning", "cross_modal_analysis"]

    def supports_modality(self, modalities: List[str]) -> bool:
        return True

    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        img_paths = request.image_paths
        query_lower = request.query.lower()

        from satquery_core.models.feature_analyzer import DynamicFeatureAnalyzer

        has_p1 = len(img_paths) > 0 and Path(img_paths[0]).exists()
        has_p2 = len(img_paths) > 1 and Path(img_paths[1]).exists()

        if ("sar" in query_lower or "radar" in query_lower or "fusion" in query_lower) and has_p1 and has_p2:
            m1 = DynamicFeatureAnalyzer.extract_scene_metrics(img_paths[0])
            m2 = DynamicFeatureAnalyzer.extract_scene_metrics(img_paths[1])
            sar_m, opt_m = (m1, m2) if m1["metadata"]["is_sar"] else (m2, m1)

            answer = (
                f"### Multi-Sensor Cross-Modal Intelligence (Optical + SAR Fusion)\n"
                f"Synergistic synthesis of **{opt_m['metadata']['sensor_type']}** and **{sar_m['metadata']['sensor_type']}**:\n\n"
                f"- **Optical Reflectance**: {opt_m['veg_pct']}% vegetation canopy, {opt_m['water_pct']}% water, {opt_m['urban_pct']}% urban infrastructure.\n"
                f"- **SAR Microwave Backscatter**: All-weather radar confirms low specular water backscatter ({sar_m['water_pct']}%) and bright double-bounce building returns ({sar_m['urban_pct']}%).\n"
                f"- **Fusion Consistency**: 94.0% spatial alignment verifying all terrain perimeters."
            )
            layman = f"Combined satellite photo and radar scan: {opt_m['veg_pct']}% greenery, {opt_m['water_pct']}% water, and {opt_m['urban_pct']}% structures verified through potential haze."
            confidence = 0.94
        elif has_p1:
            m = DynamicFeatureAnalyzer.extract_scene_metrics(img_paths[0])
            meta = m["metadata"]
            w_pct, v_pct, u_pct, s_pct = m["water_pct"], m["veg_pct"], m["urban_pct"], m["soil_pct"]
            categories = [
                ("Vegetation / Forest Canopy", v_pct),
                ("Water / Hydrological Surface", w_pct),
                ("Built-Up / Urban Infrastructure", u_pct),
                ("Bare Soil / Open Terrain", s_pct)
            ]
            categories.sort(key=lambda x: x[1], reverse=True)
            dominant_name, dominant_pct = categories[0]

            answer = (
                f"### Remote Sensing Scene Interpretation\n"
                f"Evaluation of **{meta['sensor_type']}** ({meta['width']}x{meta['height']} px) for *'{request.query}'*:\n\n"
                f"- **Dominant Land Cover**: **{dominant_name}** ({dominant_pct}% coverage).\n"
                f"- **Vegetation / Crops**: {v_pct}%\n"
                f"- **Water Bodies**: {w_pct}%\n"
                f"- **Built-Up Grid**: {u_pct}%\n"
                f"- **Bare Ground**: {s_pct}%\n\n"
                f"### Environmental & Spatial Summary\n"
                f"Scene exhibits stable radiometric signatures with clean class boundaries and zero observed anomalies."
            )
            layman = f"This image shows predominantly {dominant_name.lower()} ({dominant_pct}%), alongside {categories[1][0]} ({categories[1][1]}%) and {categories[2][0]} ({categories[2][1]}%)."
            confidence = 0.91
        else:
            answer = f"Remote sensing evaluation for '{request.query}' completed."
            layman = f"Query processed."
            confidence = 0.85

        return SpecialistOutput(
            answer=answer,
            plain_language_solution=layman,
            evidence=VisualEvidence(evidence_type="none"),
            confidence_score=confidence,
            metadata={"specialist": self.name, "task": task, "status": "evaluation"}
        )


class MockGroundingSpecialist(BaseSpecialist):
    """Spatial Grounding Specialist for dynamic object localization and bounding boxes."""

    def __init__(self):
        super().__init__(name="RS-Grounding-Head", version="Spatial-Grounding-v1.0", is_mock=True)

    def supports_task(self, task: str) -> bool:
        return task in ["grounding", "localization"]

    def supports_modality(self, modalities: List[str]) -> bool:
        return True

    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        img_paths = request.image_paths
        if not img_paths or not Path(img_paths[0]).exists():
            return SpecialistOutput(
                answer="No image available for grounding.",
                plain_language_solution="Please upload an image.",
                evidence=VisualEvidence(evidence_type="none"),
                confidence_score=0.5,
                metadata={"specialist": self.name, "task": "grounding"}
            )

        from satquery_core.models.feature_analyzer import DynamicFeatureAnalyzer

        boxes, labels, answer, layman, confidence = DynamicFeatureAnalyzer.ground_query(
            image_path=img_paths[0],
            query=request.query
        )

        evidence = VisualEvidence(
            evidence_type="bounding_boxes",
            bounding_boxes=boxes,
            labels=labels,
            spatial_metadata={"coordinate_system": "Normalized (0.0 - 1.0)", "objects_detected": len(boxes)}
        )

        return SpecialistOutput(
            answer=answer,
            plain_language_solution=layman,
            evidence=evidence,
            confidence_score=confidence,
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
            f"### Executive Summary\n"
            f"Bi-temporal Siamese change analysis identified surface transition across **{change_pct}%** of the surveyed scene.\n\n"
            f"### Surface Dynamics & Structural Classification\n"
            f"- **Spatial Extent**: {change_pct}% of total surveyed ground resolution cells show active radiometric divergence.\n"
            f"- **Dynamic Category**: MODERATE EXPANSION.\n"
            f"- **Ground Impact**: Visual evidence indicates localized structural development and ground clearing between observation dates.\n\n"
            f"### Strategic & Operational Advisory\n"
            f"Schedule re-acquisition to verify boundary stabilization and ensure ongoing environmental compliance."
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
