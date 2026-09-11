"""
SatQuery AI — Standard Data Contracts & Schemas
These schemas are shared across the API, Agent, Specialists, and Frontend.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time


class AnalysisRequest(BaseModel):
    """Incoming user request from the web interface or API."""
    query: str = Field(..., description="User natural language question or task instruction")
    image_paths: List[str] = Field(..., description="List of 1 or 2 image file paths")
    modalities: Optional[List[Literal["optical", "sar"]]] = Field(
        default=None, 
        description="Detected or user-specified modalities for each image"
    )
    timestamps: Optional[List[str]] = Field(
        default=None,
        description="Optional date/time tags for bi-temporal pairs, e.g. ['2023-01', '2024-01']"
    )
    task_override: Optional[str] = Field(
        default=None,
        description="Optional manual task override (e.g. for testing benchmarks)"
    )


class VisualEvidence(BaseModel):
    """Visual proof backing the AI's textual answer."""
    evidence_type: Literal["bounding_boxes", "change_mask", "heatmap", "overlay", "none"] = "none"
    bounding_boxes: Optional[List[List[float]]] = Field(
        default=None, 
        description="Normalized coordinates: [[ymin, xmin, ymax, xmax], ...]"
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Text labels corresponding to each bounding box"
    )
    mask_path: Optional[str] = Field(
        default=None,
        description="Path to generated binary or semantic change mask image"
    )
    preview_path: Optional[str] = Field(
        default=None,
        description="Path to generated web-renderable PNG preview for GeoTIFF or raw formats"
    )
    preview_paths: Optional[List[str]] = Field(
        default=None,
        description="Web-renderable preview paths for multi-image inputs"
    )
    spatial_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Georeferencing metadata (CRS, bounds, ground resolution in meters)"
    )


class TraceStep(BaseModel):
    """A single observable execution step in the agent pipeline."""
    step_number: int
    step_name: str
    action_taken: str
    tool_used: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    output_summary: str
    timestamp: float = Field(default_factory=time.time)


class AnalysisResult(BaseModel):
    """Final unified response returned by SatQuery AI."""
    query: str
    answer: str = Field(..., description="Clear textual response to the user's query")
    plain_language_solution: Optional[str] = Field(
        default=None,
        description="Direct layman-friendly answer without complex technical jargon"
    )
    evidence: VisualEvidence = Field(default_factory=VisualEvidence)
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Calculated confidence score between 0.0 and 1.0"
    )
    confidence_level: Literal["High", "Moderate", "Low"] = "Moderate"
    execution_trace: List[TraceStep] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = "satquery-v1.0"
