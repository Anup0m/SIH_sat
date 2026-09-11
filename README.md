# SatQuery AI 🛰️
**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**  
**Smart India Hackathon (SIH 2026) | Problem Statement: SIH26167 (ISRO / SAC)**

---

## 📌 Executive Summary

SatQuery AI is an agentic, multimodal vision-language system engineered specifically for Earth observation and remote sensing. It handles:
1. **Single Satellite Images:** Optical / multispectral or SAR radar imagery for Visual Question Answering (VQA), detailed scene captioning, and text-guided spatial grounding (bounding box localization).
2. **Bi-Temporal Image Pairs:** Two coregistered images of the same location acquired at different dates ($T_1, T_2$) to detect, describe, and map surface changes.
3. **Cross-Modal Optical + SAR Pairs:** Joint multi-sensor analysis extracting complementary features (optical surface reflectance + SAR radar structural/moisture penetration).
4. **Remote Sensing Adaptation:** Visual and vision-language components adapted on remote sensing instructions (`BigEarthNet.txt` with 9.55M instructions linking Sentinel-1 SAR and Sentinel-2 optical imagery).
5. **Agentic Orchestration & Auditability:** Input validation, automatic intent classification, specialist tool selection, sequential multi-step execution, and an observable execution trace (no black-box thinking).
6. **Verifiable Evidence & Confidence:** Real bounding boxes, change difference heatmaps, mathematically grounded confidence scores, and certified downloadable mission reports.

---

## 🚀 Quickstart: Running SatQuery AI

To launch the complete interactive workstation locally:

```powershell
# 1. Run the master launcher
python run_satquery.py
```

The workstation will automatically open in your browser at:  
👉 **`http://localhost:8000`**

---

## 📂 Project Architecture

```text
├── TASK_SHEET.md              # Master engineering task sheet tracking all 13 phases
├── run_satquery.py            # One-command root application launcher
├── configs/                   # Production hyperparameters and tool registry
│   ├── models_registry.yaml   # Registered AI specialists and capability contracts
│   ├── training_vlm.yaml      # RS-VLM adaptation config (LoRA r=16)
│   ├── training_change.yaml   # Siamese change detector config (BCE + Dice loss)
│   └── training_grounding.yaml# Spatial grounding head config
├── satquery_core/             # Core engineering package
│   ├── schemas.py             # Shared Pydantic data contracts (Request, Evidence, Trace, Result)
│   ├── data_engine/           # Remote Sensing I/O, GeoTIFF, cleaner, dataset loaders
│   │   ├── geotiff_io.py      # Optical contrast stretching & SAR dB log-scaling
│   │   ├── cleaner.py         # Automated data health & corruption filtering
│   │   └── dataset_loaders.py # PyTorch dataset loaders for BigEarthNet, VRSBench, CDVQA
│   ├── models/                # Base class and specialist tool implementations
│   │   ├── base_specialist.py # Abstract base class for all AI tools
│   │   └── mock_specialists.py# Zero-compromise offline specialists for local dev
│   ├── agent/                 # Agentic orchestration engine
│   │   ├── input_validator.py # Modality, dimension, and format verification
│   │   ├── query_parser.py    # Intent classifier & multi-step workflow planner
│   │   ├── confidence.py      # Mathematical confidence estimation
│   │   └── orchestrator.py    # Master coordinator & observable trace logger
│   └── reporting/             # Certified analytical mission report generator
├── backend/                   # FastAPI REST backend server
│   └── server.py              # Endpoints: /api/analyze, /api/validate, /api/models, /api/health
├── Dataminds/                 # Modern SatQuery AI Web Dashboard & Analysis Interface
│   ├── index.html             # High-tech landing page with turnkey demo launcher & voice input
│   ├── output.html            # Analysis workstation with split slider, evaluation trace & report
│   ├── hero.js & output.js    # Interactive controls, bounding boxes, and observable pipeline
│   └── styles.css             # Polished dark command center theme styling
├── training/                  # Uncompromising lab training pipelines (Phases 4 & 9)
│   ├── train_vlm.py           # VLM fine-tuning on BigEarthNet.txt with LoRA & checkpoints
│   ├── train_change.py        # Siamese change detection training with BCE + Dice
│   ├── evaluate_benchmarks.py # Standard evaluation metrics (Accuracy, IoU@0.5, BLEU)
│   └── launcher.py            # Unified training launcher with CUDA diagnostics
├── tests/                     # Test suites
│   ├── test_data_pipeline.py  # Phase 2 verification suite
│   └── test_full_system.py    # Master end-to-end integration test suite
└── data/                      # Dataset directories
    ├── raw/                   # Full downloaded archives (BigEarthNet, VRSBench)
    └── samples/               # Lightweight local developer slices (<1.5 GB)
```

---

## 🧪 Testing the Software System

To verify that all components, specialists, input validation, and report generators are working:

```powershell
# Run the complete end-to-end integration test suite
python tests/test_full_system.py
```

Expected output:
```text
[TEST 1] Single Image VQA Workflow...               [PASS]
[TEST 2] Text-Guided Spatial Grounding Workflow...   [PASS]
[TEST 3] Bi-Temporal Change Detection Workflow...   [PASS]
[TEST 4] Cross-Modal Optical + SAR Analysis...       [PASS]
[TEST 5] Input Validation & Edge Cases...           [PASS]
[TEST 6] Analytical Mission Report Generation...    [PASS]
ALL SATQUERY AI INTEGRATION TESTS PASSED (100% SUCCESS)
```

---

## ⚡ College Lab GPU Execution (Phase 9)

When running the heavy training runs at the college lab:

```powershell
# 1. Hardware diagnostics & evaluation
python training/launcher.py --task eval

# 2. Train RS-VLM on BigEarthNet.txt (with LoRA and automatic checkpointing)
python training/launcher.py --task vlm

# 3. Train Siamese Change Detector
python training/launcher.py --task change

# 4. Resume interrupted training anytime
python training/launcher.py --task vlm --resume
```
