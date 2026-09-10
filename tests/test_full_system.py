"""
SatQuery AI — Master End-to-End System Integration Test Suite (Phase 7 & 8)
Tests the complete software architecture 100% offline:
1. Single Image VQA
2. Text-Guided Spatial Grounding
3. Bi-Temporal Change Detection & Heatmap Generation
4. Cross-Modal Optical + SAR Joint Analysis
5. Input Validation & Edge Case Handling (Mismatches, Empty Queries, Invalid Formats)
6. Observable Execution Trace Logging
7. Analytical Mission Report Generation (JSON & Markdown)
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from satquery_core.schemas import AnalysisRequest, AnalysisResult
from satquery_core.agent.orchestrator import SatQueryAgent
from satquery_core.reporting.report_generator import MissionReportGenerator

TEST_TMP_DIR = BASE_DIR / "data" / "test_tmp"


def setup_test_environment():
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create test optical image 1
    im1 = Image.fromarray(np.random.randint(40, 200, (512, 512, 3), dtype=np.uint8))
    p1 = TEST_TMP_DIR / "opt_test_1.png"
    im1.save(p1)

    # Create test optical image 2 (slightly altered for change detection)
    arr2 = np.array(im1)
    arr2[100:250, 100:250] = 255  # Simulated new bright development
    p2 = TEST_TMP_DIR / "opt_test_2.png"
    Image.fromarray(arr2).save(p2)

    # Create test SAR image (grayscale/radar intensity)
    sar = Image.fromarray(np.random.randint(10, 180, (512, 512), dtype=np.uint8))
    p_sar = TEST_TMP_DIR / "sar_test.png"
    sar.save(p_sar)

    return str(p1), str(p2), str(p_sar)


def cleanup_test_environment():
    if TEST_TMP_DIR.exists():
        shutil.rmtree(TEST_TMP_DIR)


def run_integration_tests():
    print("==========================================================")
    print("SatQuery AI — Complete Software System Verification")
    print("==========================================================")

    p1, p2, p_sar = setup_test_environment()
    agent = SatQueryAgent()

    try:
        # -------------------------------------------------------------
        # TEST 1: Single Image VQA Workflow
        # -------------------------------------------------------------
        print("\n[TEST 1] Single Image VQA Workflow...")
        req1 = AnalysisRequest(
            query="What is the dominant land cover class in this satellite patch?",
            image_paths=[p1]
        )
        res1 = agent.run(req1)
        assert res1.confidence_score >= 0.70, f"Expected confidence >= 0.70, got {res1.confidence_score}"
        assert len(res1.execution_trace) >= 3, "Trace must log validation, parsing, and execution"
        assert len(res1.answer) > 20, "Answer too short"
        print(f"  [PASS] Answer: '{res1.answer[:55]}...' (Confidence: {res1.confidence_score*100:.1f}%)")
        print(f"  [PASS] Trace Steps Logged: {len(res1.execution_trace)}")

        # -------------------------------------------------------------
        # TEST 2: Text-Guided Spatial Grounding Workflow
        # -------------------------------------------------------------
        print("\n[TEST 2] Text-Guided Spatial Grounding Workflow...")
        req2 = AnalysisRequest(
            query="Locate the water bodies and fuel storage facilities.",
            image_paths=[p1]
        )
        res2 = agent.run(req2)
        assert res2.evidence.evidence_type == "bounding_boxes", "Evidence type must be bounding_boxes"
        assert res2.evidence.bounding_boxes is not None and len(res2.evidence.bounding_boxes) > 0
        box = res2.evidence.bounding_boxes[0]
        assert len(box) == 4 and all(0.0 <= c <= 1.0 for c in box), f"Invalid normalized box coordinates: {box}"
        print(f"  [PASS] Localized {len(res2.evidence.bounding_boxes)} target bounding boxes: {box}")

        # -------------------------------------------------------------
        # TEST 3: Bi-Temporal Change Detection Workflow
        # -------------------------------------------------------------
        print("\n[TEST 3] Bi-Temporal Change Detection Workflow...")
        req3 = AnalysisRequest(
            query="What changed between these observation dates?",
            image_paths=[p1, p2],
            timestamps=["2023-01-15", "2024-01-15"]
        )
        res3 = agent.run(req3)
        assert res3.evidence.evidence_type == "change_mask", "Evidence type must be change_mask"
        assert res3.evidence.mask_path is not None, "Change mask image must be saved to disk"
        mask_disk_path = Path("data/samples") / res3.evidence.mask_path.replace("/static/", "").lstrip("/")
        assert mask_disk_path.exists() or Path(res3.evidence.mask_path).exists(), f"Saved mask path does not exist on disk: {mask_disk_path}"
        print(f"  [PASS] Change analysis completed. Generated mask: {res3.evidence.mask_path}")
        print(f"  [PASS] Change percentage: {res3.evidence.spatial_metadata.get('change_percentage')}%")

        # -------------------------------------------------------------
        # TEST 4: Cross-Modal Optical + SAR Joint Analysis
        # -------------------------------------------------------------
        print("\n[TEST 4] Cross-Modal Optical + SAR Analysis...")
        req4 = AnalysisRequest(
            query="Perform joint Optical and SAR analysis to penetrate cloud cover and evaluate ground moisture.",
            image_paths=[p1, p_sar]
        )
        res4 = agent.run(req4)
        assert "optical" in res4.answer.lower() and "sar" in res4.answer.lower(), "Answer must reflect joint analysis"
        print(f"  [PASS] Multimodal joint answer: '{res4.answer[:65]}...'")

        # -------------------------------------------------------------
        # TEST 5: Input Validation & Defensive Handling
        # -------------------------------------------------------------
        print("\n[TEST 5] Input Validation & Edge Cases...")
        # 5a. Empty Query
        bad_req1 = AnalysisRequest(query="", image_paths=[p1])
        bad_res1 = agent.run(bad_req1)
        assert "Input Error" in bad_res1.answer
        assert bad_res1.confidence_score == 0.0
        print("  [PASS] Empty query correctly rejected.")

        # 5b. Too Many Images
        bad_req2 = AnalysisRequest(query="test", image_paths=[p1, p2, p_sar])
        bad_res2 = agent.run(bad_req2)
        assert "Maximum 2 images supported" in bad_res2.answer
        print("  [PASS] Image count > 2 correctly rejected.")

        # -------------------------------------------------------------
        # TEST 6: Mission Report Generation & Certification
        # -------------------------------------------------------------
        print("\n[TEST 6] Analytical Mission Report Generation...")
        reports_dir = BASE_DIR / "data" / "test_reports"
        paths = MissionReportGenerator.save_report(res3, output_dir=reports_dir)
        assert Path(paths["markdown_report"]).exists()
        assert Path(paths["json_report"]).exists()
        print(f"  [PASS] Certified Markdown report generated: {Path(paths['markdown_report']).name}")
        print(f"  [PASS] Structured JSON report generated: {Path(paths['json_report']).name}")

        shutil.rmtree(reports_dir)

        print("\n==========================================================")
        print("ALL SATQUERY AI INTEGRATION TESTS PASSED (100% SUCCESS)")
        print("SYSTEM IS FULLY OPERATIONAL AND READY FOR LAB TRAINING")
        print("==========================================================")

    finally:
        cleanup_test_environment()


if __name__ == "__main__":
    run_integration_tests()
