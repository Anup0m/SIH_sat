"""
SatQuery AI — Base Specialist Architecture
Every AI tool/model in SatQuery inherits from BaseSpecialist.
This ensures a strictly standardized input/output contract across all capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from satquery_core.schemas import AnalysisRequest, VisualEvidence


class SpecialistOutput:
    """Standard output returned by any specialist to the orchestrating agent."""
    def __init__(
        self,
        answer: str,
        evidence: Optional[VisualEvidence] = None,
        confidence_score: float = 0.85,
        metadata: Optional[Dict[str, Any]] = None,
        plain_language_solution: Optional[str] = None
    ):
        self.answer = answer
        self.evidence = evidence or VisualEvidence()
        self.confidence_score = confidence_score
        self.metadata = metadata or {}
        self.plain_language_solution = plain_language_solution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "plain_language_solution": self.plain_language_solution,
            "evidence": self.evidence.model_dump(),
            "confidence_score": self.confidence_score,
            "metadata": self.metadata
        }


class BaseSpecialist(ABC):
    """Abstract Base Class for all AI specialists."""

    def __init__(self, name: str, version: str, is_mock: bool = False):
        self.name = name
        self.version = version
        self.is_mock = is_mock

    @abstractmethod
    def supports_task(self, task: str) -> bool:
        """Returns True if this specialist can handle the requested task."""
        pass

    @abstractmethod
    def supports_modality(self, modalities: List[str]) -> bool:
        """Returns True if this specialist supports the given image modalities."""
        pass

    @abstractmethod
    def execute(self, request: AnalysisRequest, task: str) -> SpecialistOutput:
        """
        Executes the specialized model inference.
        Returns:
            SpecialistOutput: answer text, visual evidence, confidence score.
        """
        pass

    def load_checkpoint(self, checkpoint_path: str):
        """Loads fine-tuned weights when available."""
        pass
