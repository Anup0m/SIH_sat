"""
SatQuery AI — Agentic Orchestration Engine (Phase 5)
Orchestrates:
1. Input validation & metadata verification
2. Natural language query interpretation
3. Specialist tool selection & parameter configuration from registry
4. Sequential / multi-step tool execution
5. Visual evidence & confidence aggregation
6. Observable execution trace generation (SIH Requirement E & F)
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from satquery_core.schemas import (
    AnalysisRequest, 
    AnalysisResult, 
    VisualEvidence, 
    TraceStep
)
from satquery_core.agent.input_validator import InputValidator
from satquery_core.agent.query_parser import QueryParser
from satquery_core.models.base_specialist import BaseSpecialist, SpecialistOutput
from satquery_core.models.mock_specialists import (
    MockVLMSpecialist,
    MockGroundingSpecialist,
    MockChangeSpecialist
)


class SatQueryAgent:
    """Master agent coordinating specialists and generating observable audit traces."""

    def __init__(self):
        self.validator = InputValidator()
        self.parser = QueryParser()
        
        # Specialist registry (dynamically loads real models if checkpoints exist)
        change_ckpt = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "change_detector" / "best_change_model.pt"
        if change_ckpt.exists():
            try:
                from satquery_core.models.real_specialists import RealChangeSpecialist
                change_spec = RealChangeSpecialist(str(change_ckpt))
            except Exception:
                change_spec = MockChangeSpecialist()
        else:
            change_spec = MockChangeSpecialist()

        self.specialists: Dict[str, BaseSpecialist] = {
            "vlm_specialist": MockVLMSpecialist(),
            "grounding_specialist": MockGroundingSpecialist(),
            "change_specialist": change_spec
        }

    def register_specialist(self, key: str, specialist: BaseSpecialist):
        """Allows swapping mock models for real fine-tuned models seamlessly."""
        self.specialists[key] = specialist

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        start_time = time.time()
        trace: List[TraceStep] = []
        step_idx = 1

        # -------------------------------------------------------------
        # STEP 1: Input Validation
        # -------------------------------------------------------------
        is_valid, val_msg, img_meta = self.validator.validate_request(request)
        trace.append(TraceStep(
            step_number=step_idx,
            step_name="Input Validation",
            action_taken="Validated image presence, file formats, and spatial compatibility.",
            output_summary=f"Status: {'PASSED' if is_valid else 'FAILED'} - {val_msg}",
            parameters={"images_count": len(request.image_paths), "metadata": img_meta}
        ))
        step_idx += 1

        if not is_valid:
            elapsed = (time.time() - start_time) * 1000.0
            return AnalysisResult(
                query=request.query,
                answer=f"Input Error: {val_msg}",
                confidence_score=0.0,
                confidence_level="Low",
                execution_trace=trace,
                processing_time_ms=round(elapsed, 2)
            )

        # -------------------------------------------------------------
        # STEP 2: Query Parsing & Task Routing
        # -------------------------------------------------------------
        plan = self.parser.parse(request.query, num_images=len(request.image_paths))
        trace.append(TraceStep(
            step_number=step_idx,
            step_name="Task Classification & Workflow Planning",
            action_taken=f"Classified query intent into primary task: '{plan['primary_task']}'.",
            output_summary=f"Planned workflow: {' -> '.join(plan['workflow'])} | {plan['reasoning']}",
            parameters={"is_multistep": plan["is_multistep"], "primary_task": plan["primary_task"]}
        ))
        step_idx += 1

        # -------------------------------------------------------------
        # STEP 3 & 4: Specialist Execution
        # -------------------------------------------------------------
        answers = []
        combined_evidence = VisualEvidence()
        confidence_scores = []

        for tool_name in plan["workflow"]:
            specialist = self.specialists.get(tool_name)
            if not specialist:
                # Fallback to VLM
                specialist = self.specialists["vlm_specialist"]
                tool_name = "vlm_specialist"

            t_start = time.time()
            out: SpecialistOutput = specialist.execute(request, task=plan["primary_task"])
            t_dur = (time.time() - t_start) * 1000.0

            answers.append(out.answer)
            confidence_scores.append(out.confidence_score)

            # Merge evidence if produced
            if out.evidence and out.evidence.evidence_type != "none":
                combined_evidence = out.evidence

            trace.append(TraceStep(
                step_number=step_idx,
                step_name=f"Specialist Execution: {specialist.name}",
                action_taken=f"Dispatched subtask to specialist '{specialist.name}' ({specialist.version}).",
                tool_used=specialist.name,
                parameters={"task": plan["primary_task"], "execution_ms": round(t_dur, 2)},
                output_summary=f"Generated output with confidence {out.confidence_score*100:.1f}%."
            ))
            step_idx += 1

        # -------------------------------------------------------------
        # STEP 5: Confidence & Result Aggregation
        # -------------------------------------------------------------
        final_answer = " ".join(answers)
        avg_confidence = float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.85
        
        conf_level = "High" if avg_confidence >= 0.85 else ("Moderate" if avg_confidence >= 0.65 else "Low")

        trace.append(TraceStep(
            step_number=step_idx,
            step_name="Result Aggregation",
            action_taken="Consolidated textual analysis, visual evidence, and calibrated overall confidence.",
            output_summary=f"Final confidence: {avg_confidence*100:.1f}% ({conf_level}). Evidence type: {combined_evidence.evidence_type}.",
            parameters={"confidence_level": conf_level, "evidence_type": combined_evidence.evidence_type}
        ))

        elapsed = (time.time() - start_time) * 1000.0
        return AnalysisResult(
            query=request.query,
            answer=final_answer,
            evidence=combined_evidence,
            confidence_score=round(avg_confidence, 2),
            confidence_level=conf_level,
            execution_trace=trace,
            processing_time_ms=round(elapsed, 2)
        )
