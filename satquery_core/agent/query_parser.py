"""
SatQuery AI — Query Parsing & Task Intent Classifier (Phase 5)
Analyzes natural language queries and maps them to specialist tasks and workflows.
"""

from typing import Dict, Any, List, Tuple


class QueryParser:
    """Interprets user intent and determines single or multi-step execution workflows."""

    GROUNDING_KEYWORDS = [
        "where is", "where are", "locate", "find", "box", "detect", "bound", "highlight",
        "show me the location", "point out"
    ]
    CHANGE_KEYWORDS = [
        "what changed", "change", "difference", "expanded", "disappeared", "new building",
        "between these dates", "over time", "before and after", "temporal"
    ]
    CROSS_MODAL_KEYWORDS = [
        "sar", "radar", "optical and sar", "optical + sar", "sar-optical", "sar + optical",
        "multispectral and radar", "penetrate clouds", "all-weather", "joint analysis",
        "complementary", "fusion", "cross-modal", "cross modal", "dielectric"
    ]
    CAPTION_KEYWORDS = [
        "describe", "caption", "overview", "summarize", "what does this image show",
        "scene description", "tell me about this scene"
    ]

    def parse(self, query: str, num_images: int) -> Dict[str, Any]:
        """
        Classifies task intent based on query semantics and input imagery.
        """
        q = query.lower().strip()
        is_pair = (num_images == 2)

        wants_cross_modal = any(k in q for k in self.CROSS_MODAL_KEYWORDS)
        wants_change_explicit = any(k in q for k in self.CHANGE_KEYWORDS)
        wants_grounding = any(k in q for k in self.GROUNDING_KEYWORDS)
        wants_caption = any(k in q for k in self.CAPTION_KEYWORDS)

        # 1. Optical + SAR Cross-Modal Fusion (highest precedence when SAR/radar/fusion is requested)
        if wants_cross_modal:
            return {
                "primary_task": "optical_sar_fusion",
                "is_multistep": False,
                "workflow": ["vlm_specialist"],
                "reasoning": "Query requires joint optical and SAR cross-modal analysis on multimodal sensor imagery."
            }

        # 2. Multi-step change + grounding (e.g. "what changed and locate the new buildings")
        if is_pair and wants_change_explicit and wants_grounding:
            return {
                "primary_task": "change_detection",
                "is_multistep": True,
                "workflow": ["change_specialist", "grounding_specialist"],
                "reasoning": "Query asks for both change analysis and spatial localization on a bi-temporal pair."
            }

        # 3. Bi-temporal change detection
        if is_pair and (wants_change_explicit or not (wants_grounding or wants_caption)):
            return {
                "primary_task": "change_detection",
                "is_multistep": False,
                "workflow": ["change_specialist"],
                "reasoning": "Bi-temporal image pair detected for change analysis."
            }

        if wants_grounding:
            return {
                "primary_task": "grounding",
                "is_multistep": False,
                "workflow": ["grounding_specialist"],
                "reasoning": "Query requests object localization / bounding boxes."
            }

        if wants_caption:
            return {
                "primary_task": "captioning",
                "is_multistep": False,
                "workflow": ["vlm_specialist"],
                "reasoning": "Query requests full scene description / caption."
            }

        # Default fallback to single-image VQA
        return {
            "primary_task": "vqa",
            "is_multistep": False,
            "workflow": ["vlm_specialist"],
            "reasoning": "Standard remote sensing visual question answering query."
        }
