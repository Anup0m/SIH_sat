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

TEST_DATASET_DIR = BASE_DIR / "data" / "test_dataset"
TEST_DATASET_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = BASE_DIR / "Dataminds"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/demo_samples", StaticFiles(directory=str(DEMO_DIR)), name="demo_samples")
app.mount("/test_dataset", StaticFiles(directory=str(TEST_DATASET_DIR)), name="test_dataset")
app.mount("/static_ui", StaticFiles(directory=str(FRONTEND_DIR)), name="static_ui")

# Instantiate Agent
agent = SatQueryAgent()


@app.get("/")
def serve_ui():
    """Serves the SatQuery AI Landing Page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/output.html")
def serve_output_page():
    """Serves the SatQuery AI Analysis Dashboard."""
    return FileResponse(FRONTEND_DIR / "output.html")


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


@app.get("/api/test_dataset")
def get_test_dataset():
    """Returns curated testing dataset catalog with sample queries for laptops."""
    return {
        "optical": [
            {
                "filename": "optical_sentinel2_bahamas_water.jpg",
                "path": "/test_dataset/optical/optical_sentinel2_bahamas_water.jpg",
                "sensor": "Sentinel-2 True Color (VNIR)",
                "features": "Shallow tropical ocean water, coral channels, deep water boundary",
                "sample_query": "Locate and map all water bodies in this scene."
            },
            {
                "filename": "optical_sentinel2_congo_forest.jpg",
                "path": "/test_dataset/optical/optical_sentinel2_congo_forest.jpg",
                "sensor": "Sentinel-2 True Color (VNIR)",
                "features": "Dense tropical rainforest canopy and river clearings",
                "sample_query": "Where are the green vegetation and forest zones?"
            },
            {
                "filename": "optical_sentinel2_minsk_urban.jpg",
                "path": "/test_dataset/optical/optical_sentinel2_minsk_urban.jpg",
                "sensor": "Sentinel-2 True Color (VNIR)",
                "features": "Metropolitan urban fabric, road network, and water reservoir",
                "sample_query": "Locate the water bodies and urban areas."
            },
            {
                "filename": "optical_sentinel2_south_georgia_coastal.jpg",
                "path": "/test_dataset/optical/optical_sentinel2_south_georgia_coastal.jpg",
                "sensor": "Sentinel-2 True Color (VNIR)",
                "features": "Deep marine ocean water, coastal boundary, rugged topography",
                "sample_query": "Where is the water coastline located?"
            }
        ],
        "geotiff": [
            {
                "filename": "landsat_rgb.tif",
                "path": "/test_dataset/geotiff/landsat_rgb.tif",
                "sensor": "Landsat-8 Multispectral GeoTIFF (.tif)",
                "features": "Coastal estuary, sediment plumes, and land cover",
                "sample_query": "Locate water bodies in this GeoTIFF."
            },
            {
                "filename": "landsat_multispectral_urban.tif",
                "path": "/test_dataset/geotiff/landsat_multispectral_urban.tif",
                "sensor": "Landsat-8 True-Color 3-Band GeoTIFF (.tif)",
                "features": "Coastal wetlands, ocean water boundary, and urban fringe",
                "sample_query": "Map the water bodies and coastal perimeter in this Landsat GeoTIFF."
            },
            {
                "filename": "sentinel2_sample.tif",
                "path": "/test_dataset/geotiff/sentinel2_sample.tif",
                "sensor": "Sentinel-2 Multispectral GeoTIFF (.tif, 16-bit)",
                "features": "Copernicus Level-2A surface reflectance",
                "sample_query": "What is the dominant land cover in this GeoTIFF?"
            },
            {
                "filename": "sentinel2_cloud_optimized_cog.tif",
                "path": "/test_dataset/geotiff/sentinel2_cloud_optimized_cog.tif",
                "sensor": "Sentinel-2 Cloud-Optimized GeoTIFF (COG, 16-bit)",
                "features": "Calibrated surface reflectance, coastal and inland water features",
                "sample_query": "Detect water channels and delineate boundaries in this COG."
            },
            {
                "filename": "global_earth_observation.tif",
                "path": "/test_dataset/geotiff/global_earth_observation.tif",
                "sensor": "Global Earth Observation True-Color GeoTIFF (.tif)",
                "features": "Continental land masses, global ocean basins, atmospheric cloud bands",
                "sample_query": "Analyze water vs land percentages across the global footprint."
            },
            {
                "filename": "elevation_shade_terrain.tif",
                "path": "/test_dataset/geotiff/elevation_shade_terrain.tif",
                "sensor": "Digital Elevation Model / Shaded Relief GeoTIFF (.tif)",
                "features": "Topographical ridge lines, valleys, drainage depressions",
                "sample_query": "Where are the low-elevation drainage basins and shadows located?"
            },
            {
                "filename": "sentinel2_minsk_urban.tif",
                "path": "/test_dataset/geotiff/sentinel2_minsk_urban.tif",
                "sensor": "Sentinel-2 Multispectral Urban GeoTIFF (.tif)",
                "features": "Metropolitan urban fabric, street grid, and water reservoir",
                "sample_query": "Locate urban buildings and water bodies in this GeoTIFF."
            },
            {
                "filename": "sentinel2_congo_rainforest.tif",
                "path": "/test_dataset/geotiff/sentinel2_congo_rainforest.tif",
                "sensor": "Sentinel-2 Multispectral Forest GeoTIFF (.tif)",
                "features": "Dense rainforest vegetation canopy, river clearing, and green canopy",
                "sample_query": "Where are the dense forest zones and waterways?"
            },
            {
                "filename": "sentinel2_bahamas_ocean_water.tif",
                "path": "/test_dataset/geotiff/sentinel2_bahamas_ocean_water.tif",
                "sensor": "Sentinel-2 Coastal Marine GeoTIFF (.tif)",
                "features": "Shallow tropical ocean water, coral channels, deep water boundary",
                "sample_query": "Locate and map all water bodies in this GeoTIFF."
            },
            {
                "filename": "sentinel2_south_georgia_coast.tif",
                "path": "/test_dataset/geotiff/sentinel2_south_georgia_coast.tif",
                "sensor": "Sentinel-2 Coastal GeoTIFF (.tif)",
                "features": "Deep marine ocean water, coastline, and rugged topography",
                "sample_query": "Where is the water coastline located?"
            }
        ],
        "sar": [
            {
                "filename": "sentinel1_sar_dual_polarization.tif",
                "path": "/test_dataset/sar/sentinel1_sar_dual_polarization.tif",
                "sensor": "Sentinel-1 SAR C-Band Dual-Polarization GeoTIFF (.tif)",
                "features": "High-resolution radar backscatter (specular water, double-bounce urban)",
                "sample_query": "Analyze radar backscatter to identify water bodies and urban structures."
            },
            {
                "filename": "sentinel1_sar_coastal_radar.tif",
                "path": "/test_dataset/sar/sentinel1_sar_coastal_radar.tif",
                "sensor": "Sentinel-1 SAR Coastal Radar GeoTIFF (.tif)",
                "features": "C-Band SAR radar coastal boundary and calm water detection",
                "sample_query": "Evaluate radar backscatter intensity and identify low-return water zones."
            },
            {
                "filename": "sentinel1_sar_agricultural_radar.tif",
                "path": "/test_dataset/sar/sentinel1_sar_agricultural_radar.tif",
                "sensor": "Sentinel-1 SAR Agricultural Radar GeoTIFF (.tif)",
                "features": "C-Band SAR radar soil moisture and crop surface roughness",
                "sample_query": "Analyze surface roughness and radar intensity across agricultural plots."
            },
            {
                "filename": "sar_sentinel1_dual_polarization.jpg",
                "path": "/test_dataset/sar/sar_sentinel1_dual_polarization.jpg",
                "sensor": "Sentinel-1 SAR C-Band Microwave (Full Composite)",
                "features": "Dual-polarization radar backscatter (specular water, double bounce)",
                "sample_query": "Analyze water boundaries using radar backscatter."
            }
        ]
    }


@app.get("/api/download_test_dataset")
def download_test_dataset():
    """Downloads the complete curated satellite test dataset as a ZIP archive."""
    zip_path = TEST_DATASET_DIR / "Satellite_Test_Images.zip"
    if not zip_path.exists():
        alt_zip = Path(r"C:\Users\Yash\Desktop\Satellite_Test_Images.zip")
        if alt_zip.exists():
            shutil.copy2(alt_zip, zip_path)
    if zip_path.exists():
        return FileResponse(
            path=str(zip_path),
            filename="Satellite_Test_Images.zip",
            media_type="application/zip"
        )
    raise HTTPException(status_code=404, detail="Test dataset zip not found.")


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

    from satquery_core.models.feature_analyzer import DynamicFeatureAnalyzer

    saved_paths = []
    preview_web_paths = []

    for img in images:
        file_dest = UPLOAD_DIR / f"{int(time.time()*1000)}_{img.filename}"
        with open(file_dest, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_paths.append(str(file_dest))

        # Check if file is TIFF (by extension or magic bytes)
        is_tif = file_dest.suffix.lower() in [".tif", ".tiff"]
        if not is_tif:
            try:
                with open(file_dest, "rb") as f_head:
                    h_bytes = f_head.read(4)
                    if h_bytes.startswith(b"II*\x00") or h_bytes.startswith(b"MM\x00*"):
                        is_tif = True
            except Exception:
                pass

        if is_tif:
            preview_name = f"preview_{file_dest.stem}.png"
            preview_file = UPLOAD_DIR / preview_name
            try:
                DynamicFeatureAnalyzer.generate_web_preview(file_dest, preview_file)
                preview_web_paths.append(f"/uploads/{preview_name}")
            except Exception as e_prev:
                print(f"[!] Warning: Could not generate web preview for {file_dest.name}: {e_prev}")
                preview_web_paths.append(f"/uploads/{file_dest.name}")
        else:
            preview_web_paths.append(f"/uploads/{file_dest.name}")

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

    # Attach generated web previews for browser display
    if preview_web_paths:
        if not result.evidence:
            result.evidence = VisualEvidence()
        result.evidence.preview_path = preview_web_paths[0]
        result.evidence.preview_paths = preview_web_paths

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


# Mount Dataminds static files for root-level asset serving (styles.css, hero.js, output.js, satellite_map.jpg)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("Starting SatQuery AI Backend on http://localhost:8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
