# SATQUERY AI — MASTER SIH CROSS-CHECK CHECKLIST (SIH26167)
*Authoritative Source of Truth & Progress Verification*

> **The Golden Rule**: A checkbox is only marked complete when the feature actually works, is tested, and can be demonstrated.

---

# PHASE 1 — REQUIREMENTS

### SIH understanding
* [x] Problem statement ID 26167 confirmed
* [x] Final objective written clearly in README.md and implementation plan
* [x] Single-image requirement understood (VQA + Grounding / Captioning)
* [x] Bi-temporal requirement understood (Change Detection, Mapping & VQA)
* [x] Optical + SAR requirement understood (Cross-modal complementary analysis)
* [x] Remote-sensing adaptation requirement understood (BigEarthNet.txt instruction tuning)
* [x] Agentic orchestration requirement understood (Automated routing & sequencing)
* [x] Evidence requirement understood (Bounding boxes, difference heatmaps)
* [x] Confidence requirement understood (Mathematical calibration, no fake percentages)
* [x] Execution-summary requirement understood (Observable step-by-step trace)
* [x] GUI/web-app requirement understood (Interactive Command Center)
* [x] Downloadable-report requirement understood (Markdown, JSON, GeoJSON)
* [x] Public benchmark requirement understood (BigEarthNet.txt, VRSBench, CDVQA)
* [x] ISRO/SAC evaluation requirement understood (Pre-georeferenced Cartosat-2S + RISAT pairs)

### Input scope
* [x] Single optical/multispectral image supported (`satquery_core/agent/input_validator.py`)
* [x] Single SAR image supported (`satquery_core/data_engine/geotiff_io.py`)
* [x] Optical + SAR pair supported
* [x] Bi-temporal pair supported
* [x] GeoTIFF supported (`geotiff_io.py`)
* [x] TIFF supported
* [x] PNG/JPEG handled only where allowed by SIH

### Final acceptance
* [x] A written requirements document exists (`TASK_SHEET.md`, `README.md`)
* [x] Every SIH requirement has a corresponding software/ML component
* [x] No SIH requirement is only “planned”

---

# PHASE 2 — DATASET ACQUISITION

## BigEarthNet.txt
* [x] Dataset source verified (`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` on HF)
* [x] Dataset downloaded (`data/raw/bigearthnet/BigEarthNet.txt.parquet` - 467 MB)
* [x] Files are complete (All 9,553,962 instruction-answer rows verified)
* [x] Sentinel-1 SAR data identified (`s1_name` column)
* [x] Sentinel-2 multispectral data identified (`patch_id` column)
* [x] Text annotations identified (`input` and `output` columns)
* [x] Image-text relationships understood
* [x] Optical/SAR pairing understood (Same tile ID, co-registered)
* [x] Dataset size measured (467 MB text metadata; full images ~100GB in lab)
* [x] Storage requirement measured
* [x] Dataset documentation saved

## VRSBench
* [x] Dataset downloaded (`data/raw/vrsbench/`)
* [x] Images identified (512x512 RGB satellite imagery)
* [x] Caption data identified (`VRSBench_EVAL_Cap.json` - 9,350 entries)
* [x] VQA data identified (`VRSBench_EVAL_vqa.json` - 37,409 entries)
* [x] Grounding/reference data identified (`VRSBench_EVAL_referring.json` - 16,159 entries)
* [x] Annotation format understood (Normalized 4-corner coordinates & boxes)
* [x] Dataset size measured (24.4 MB eval JSONs; full image archive 12.5GB in lab)

## RSVQA & CDVQA
* [x] Dataset sources and splits identified
* [x] Image-pair and Question-Answer structures understood
* [x] Siamese difference loader written for bi-temporal pairs

### Data verification
* [x] Random samples can be opened (`tests/test_data_pipeline.py`)
* [x] Broken files detected (`satquery_core/data_engine/cleaner.py`)
* [x] Missing annotations detected
* [x] Duplicate/problematic data identified (Black tiles, cloud oversaturation)
* [x] Dataset paths are configurable (Parameterized via YAML configs)
* [x] Dataset locations are NOT hardcoded to one computer

---

# PHASE 3 — DATA PREPARATION

* [x] Loader written (`satquery_core/data_engine/dataset_loaders.py`)
* [x] Preprocessing written (`geotiff_io.py`)
* [x] Dataset validation written (`cleaner.py`)
* [x] Train split prepared (`BigEarthNetDataset`, `VRSBenchDataset`)
* [x] Validation split prepared
* [x] Sample visualization tool created (`frontend/index.html` & `tests/test_data_pipeline.py`)
* [x] Dataset statistics generated (9.55M rows inspected)
* [x] Corrupted samples handled (Automatic rejection in `cleaner.py`)
* [x] Missing annotations handled
* [x] Dataset configuration files created (`configs/training_*.yaml`)

### Image preprocessing
* [x] Optical images handled correctly (2nd-98th percentile contrast stretch)
* [x] SAR images handled correctly ($10 \log_{10}(\text{power})$ decibel scaling)
* [x] Multispectral bands handled correctly
* [x] Image sizes standardized where required (512x512 / 256x256)
* [x] Normalization defined (Float 0.0-1.0 and Uint8 0-255)
* [x] Channels handled correctly
* [x] GeoTIFF reading works
* [x] Metadata can be preserved/read
* [x] Pair alignment information can be preserved

### Training data verification
* [x] One sample can go through the complete preprocessing pipeline
* [x] One batch can be loaded successfully (`tests/test_data_pipeline.py` passed)
* [x] Batch contents visually verified
* [x] No accidental image/annotation mismatch

---

# PHASE 4 — MODEL DESIGN

| Capability | Actual Model Architecture | Target Dataset | Train / Fine-tune? | Output Format |
| :--- | :--- | :--- | :--- | :--- |
| **Remote-Sensing Adaptation** | Qwen2-VL / Vision-Language Model | BigEarthNet.txt | LoRA Adaptation ($r=16, \alpha=32$) | Natural language answer |
| **Single-Image VQA** | RS-VLM Specialist | VRSBench / RSVQA | LoRA Fine-Tuned | Text answer + Confidence |
| **Grounding / Captioning** | Spatial Box Head / VLM | VRSBench | Fine-Tuned Regressor | Bounding boxes $[ymin, xmin, ymax, xmax]$ |
| **Bi-Temporal Change** | Siamese Difference Network | CDVQA / SECOND | Trained (BCE + Dice Loss) | Binary change mask ($512 \times 512$) + Summary |
| **Cross-Modal Optical + SAR** | Multimodal RS-VLM | BigEarthNet S1+S2 | Joint Instruction Tuned | Complementary textual & spatial evidence |

* [x] Every mandatory capability has an actual implementation
* [x] Shared-model opportunities identified (VLM handles VQA, captioning, and joint queries)
* [x] Unnecessary duplicate models avoided
* [x] No model exists without a clear purpose

---

# PHASE 5 — TRAINING PIPELINES (Pre-GPU Provisioning)

* [x] Training environment file created (`requirements.txt`)
* [x] Dependencies documented
* [x] Dataset paths configurable (`configs/training_*.yaml`)
* [x] Model paths configurable
* [x] GPU selection configurable (Automatic CUDA/CPU detection)
* [x] Batch size configurable
* [x] Gradient accumulation supported where needed
* [x] Mixed precision supported where appropriate
* [x] Checkpoint saving implemented (`torch.save` state dicts)
* [x] Training resume implemented (`--resume` flag)
* [x] Best-model saving implemented
* [x] Final-model saving implemented
* [x] Logs saved
* [x] Training metrics saved
* [x] Turnkey Google Colab / Kaggle free GPU notebook created (`notebooks/satquery_free_gpu_training.ipynb`)
* [x] Linux lab environment setup script created (`scripts/setup_lab_gpu.sh`)
* [x] Windows lab batch setup script created (`scripts/setup_lab_windows.bat`)

---

# PHASE 6 — BACKEND

* [x] Backend starts successfully (`backend/server.py` on FastAPI/Uvicorn)
* [x] Configuration system works
* [x] Logging works
* [x] Error handling works (HTTP 400 with descriptive detail)
* [x] API documentation available (Interactive Swagger at `/docs`)
* [x] Single image upload (`/api/analyze`)
* [x] Multiple image upload (Bi-temporal and Optical+SAR pairs)
* [x] GeoTIFF reading
* [x] TIFF reading
* [x] Metadata extraction
* [x] Temporary-file management (`data/uploads/`)
* [x] Invalid-file handling

---

# PHASE 7 — INPUT VALIDATION

* [x] Number of images validated (1 or 2 accepted; $>2$ rejected)
* [x] Image format validated (.png, .jpg, .jpeg, .tif, .tiff)
* [x] Image modality detected
* [x] Pair compatibility checked
* [x] Dimension matching verified between pairs
* [x] Empty queries rejected with informative error
* [x] Bad inputs handled defensively without crashing

---

# PHASE 8 — MODEL REGISTRY

* [x] Registry loads successfully (`configs/models_registry.yaml`)
* [x] Tools defined: VLM Specialist, Grounding Specialist, Siamese Change Specialist
* [x] Supported inputs, modalities, and parameter constraints recorded
* [x] Agent queries registry dynamically
* [x] Dynamic swapping between mock and real trained checkpoints verified live

---

# PHASE 9 — AGENTIC ORCHESTRATION

* [x] Agent receives user question (`satquery_core/agent/orchestrator.py`)
* [x] Agent parses task intent (`satquery_core/agent/query_parser.py`)
* [x] Agent considers image count and sensor modality
* [x] VQA query routes correctly
* [x] Grounding query routes correctly
* [x] Captioning query routes correctly
* [x] Change query routes correctly
* [x] Optical + SAR query routes correctly
* [x] Multi-step execution supported (e.g. Change Detection $\rightarrow$ Spatial Grounding)
* [x] Observable execution trace generated for every step

---

# PHASE 10 TO 15 — OUTPUTS, EVIDENCE, CONFIDENCE & REPORTS

* [x] Standardized specialist output schema (`SpecialistOutput`)
* [x] Grounding returns normalized bounding boxes $[ymin, xmin, ymax, xmax]$
* [x] Change returns actual difference heatmap image on disk + surface percentage
* [x] Visual evidence visually displayed on UI
* [x] No fake evidence
* [x] Mathematical confidence estimation (`satquery_core/agent/confidence.py`)
* [x] Confidence categories defined: High, Moderate, Low
* [x] Observable execution trace table generated (SIH Requirement E & F)
* [x] Certified Analytical Mission Report generated (JSON, Markdown, RFC 7946 GeoJSON)

---

# PHASE 16 TO 18 — FRONTEND & FULL-SYSTEM INTEGRATION

* [x] High-tech dark command-center UI (`frontend/index.html`, `styles.css`)
* [x] Drag-and-drop satellite imagery uploader
* [x] Preset query chips
* [x] Bounding box canvas overlay
* [x] Interactive split-view Before/After change swipe slider
* [x] Live observable execution trace timeline widget
* [x] One-click certified mission report download
* [x] Master end-to-end integration test suite passed 100% (`tests/test_full_system.py`)
* [x] System runs 100% offline without external cloud dependencies

---

# PHASE 19 TO 25 — COLLEGE LAB GPU COMPUTE & FINAL BENCHMARK

* [ ] Full-scale multi-epoch VLM adaptation on complete BigEarthNet.txt image archive
* [ ] Full-scale Siamese change network convergence on complete CDVQA/SECOND split
* [ ] Full evaluation on complete 37,409 VRSBench test questions
* [ ] ISRO/SAC Cartosat-2S + RISAT final validation test run
* [ ] Final live demonstration rehearsal
