"""
SatQuery AI — FastAPI Backend Server (Phase 5)
Provides high-performance REST endpoints for:
1. /api/health: System diagnostics and specialist status
2. /api/models: Registered models and capability metadata
3. /api/analyze: Unified multimodal image analysis with observable execution traces
4. /api/report: Mission report generation (JSON / Markdown / PDF export)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional
import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from satquery_core.schemas import AnalysisRequest, AnalysisResult
from satquery_core.agent.orchestrator import SatQueryAgent
from satquery_core.reporting.report_generator import MissionReportGenerator

# Initialize FastAPI
app = FastAPI(
    title="SatQuery AI — Multimodal Remote Sensing Assistant API",
    description="Backend service powering natural-language satellite image analysis for SIH26167.",
    version="1.0.0"
)

# CORS Middleware (allows local frontend connections without restrictions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

# Storage directories
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = BASE_DIR / "data" / "samples"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

DEMO_DIR = BASE_DIR / "data" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/demo_samples", StaticFiles(directory=str(DEMO_DIR)), name="demo_samples")
app.mount("/static_ui", StaticFiles(directory=str(FRONTEND_DIR)), name="static_ui")

# Instantiate Agent
agent = SatQueryAgent()


@app.get("/")
def serve_ui():
    """Serves the SatQuery AI Interactive Web Dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check():
    """Returns system status, active specialist count, and hardware info."""
    import torch
    cuda_avail = torch.cuda.is_available() if "torch" in sys.modules else False
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "engine_version": "satquery-v1.0",
        "hardware": {
            "cuda_available": cuda_avail,
            "gpu_device": gpu_name
        },
        "registered_specialists": list(agent.specialists.keys()),
        "mode": "Active (Zero-Compromise Offline Readiness)"
    }


@app.get("/api/models")
def get_models():
    """Returns available tools and permitted parameters from model registry."""
    reg_path = BASE_DIR / "configs" / "models_registry.yaml"
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"specialists": {k: {"name": v.name, "version": v.version} for k, v in agent.specialists.items()}}


@app.get("/api/demos")
def get_demo_cases():
    """Returns turnkey SIH demo test cases for 1-click evaluation."""
    return [
        {
            "id": "grounding",
            "title": "📍 Grounding & VQA",
            "badge": "Single Scene",
            "description": "Localize water reservoir & facility structures",
            "query": "Where are the water bodies located in this scene?",
            "images": ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"],
            "timestamps": ""
        },
        {
            "id": "change",
            "title": "🔄 Bi-Temporal Change",
            "badge": "T1 vs T2 Pair",
            "description": "Siamese difference detection & split comparison slider",
            "query": "What changed between these observation dates?",
            "images": ["/demo_samples/UseCase3_Change_Before.jpg", "/demo_samples/UseCase3_Change_After.jpg"],
            "timestamps": "2023-01-15, 2024-01-15"
        },
        {
            "id": "optical_sar",
            "title": "⚡ Optical + SAR",
            "badge": "Cross-Modal",
            "description": "Joint optical reflectance and SAR radar backscatter analysis",
            "query": "Perform joint Optical and SAR analysis to classify land cover.",
            "images": ["/demo_samples/UseCase4_Fusion_Optical.jpg", "/demo_samples/UseCase4_Fusion_SAR.jpg"],
            "timestamps": "2023-08-10, 2023-08-10"
        },
        {
            "id": "captioning",
            "title": "📝 Scene Captioning",
            "badge": "VLM Synthesis",
            "description": "Multi-sensor terrain, agriculture & infrastructure summary",
            "query": "Describe the dominant terrain and infrastructure.",
            "images": ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"],
            "timestamps": ""
        }
    ]


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_imagery(
    query: str = Form(..., description="User question or instruction"),
    images: List[UploadFile] = File(..., description="1 or 2 satellite imagery files"),
    timestamps: Optional[str] = Form(None, description="Comma-separated dates, e.g. '2023-01,2024-01'"),
    task_override: Optional[str] = Form(None, description="Optional manual task override")
):
    """
    Primary analysis endpoint:
    Accepts 1 or 2 satellite images + query -> executes agent orchestration -> returns answer, visual evidence, confidence & trace.
    """
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least 1 satellite image must be uploaded.")
    if len(images) > 2:
        raise HTTPException(status_code=400, detail="Maximum 2 images supported per analysis session.")

    saved_paths = []
    for img in images:
        file_dest = UPLOAD_DIR / f"{int(time.time()*1000)}_{img.filename}"
        with open(file_dest, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_paths.append(str(file_dest))

    ts_list = [t.strip() for t in timestamps.split(",")] if timestamps else None

    # Construct request
    req = AnalysisRequest(
        query=query,
        image_paths=saved_paths,
        timestamps=ts_list,
        task_override=task_override
    )

    # Execute agent
    result = agent.run(req)

    # Normalize mask path to a web accessible URL
    if result.evidence and result.evidence.mask_path:
        mp = result.evidence.mask_path.replace("\\", "/")
        if "data/samples/" in mp:
            mp = mp.split("data/samples/")[-1]
        if not mp.startswith("/"):
            mp = f"/{mp}"
        if not mp.startswith("/static/"):
            mp = f"/static{mp}" if mp.startswith("/output_masks") else mp
        result.evidence.mask_path = mp

    # Automatically persist mission report
    try:
        MissionReportGenerator.save_report(result, output_dir=REPORTS_DIR)
    except Exception as e:
        print(f"[!] Warning: Could not auto-save report: {e}")

    return result


@app.post("/api/report/export")
def export_report(result: AnalysisResult):
    """Generates and returns markdown/json report files."""
    paths = MissionReportGenerator.save_report(result, output_dir=REPORTS_DIR)
    return {
        "status": "Report Generated",
        "paths": paths
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting SatQuery AI Backend on http://localhost:8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
