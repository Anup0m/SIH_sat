// SatQuery AI — Frontend Controller (Phase 6 & 12)
// Coordinates drag-and-drop, 1-click SIH test suites, multimodal API calls, visual evidence rendering, and observable traces.

const API_BASE = window.location.origin;

let selectedFiles = [];
let latestAnalysisResult = null;

// DOM Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const filePreviews = document.getElementById("file-previews");
const temporalGroup = document.getElementById("temporal-group");
const timestampsInput = document.getElementById("timestamps-input");
const queryInput = document.getElementById("query-input");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");

// Viewer Elements
const emptyViewer = document.getElementById("empty-viewer");
const singleWrapper = document.getElementById("single-image-wrapper");
const mainDisplayImg = document.getElementById("main-display-img");
const bboxOverlay = document.getElementById("bbox-overlay");
const comparisonWrapper = document.getElementById("comparison-wrapper");
const imgBefore = document.getElementById("img-before");
const imgAfter = document.getElementById("img-after");
const afterSliderWrapper = document.getElementById("after-slider-wrapper");
const compareSlider = document.getElementById("compare-slider");
const sliderLine = document.getElementById("slider-line");
const toggleMaskBtn = document.getElementById("toggle-mask-btn");
const evidenceTag = document.getElementById("evidence-tag");
const tagAfter = document.getElementById("tag-after");

// HUD Telemetry Elements
const hudConf = document.getElementById("hud-conf");
const hudTime = document.getElementById("hud-time");
const hudTask = document.getElementById("hud-task");
const hudEvidence = document.getElementById("hud-evidence");

// Results Elements
const answerContainer = document.getElementById("answer-container");
const confBadge = document.getElementById("conf-badge");
const traceTimeline = document.getElementById("trace-timeline");
const downloadReportBtn = document.getElementById("download-report-btn");

// Demo Presets Map
const DEMO_PRESETS = {
  grounding: {
    title: "VQA & Spatial Grounding",
    query: "Where are the water bodies located in this scene?",
    files: ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"],
    timestamps: "",
    task: "Grounding"
  },
  change: {
    title: "Bi-Temporal Change Detection",
    query: "What changed between these observation dates?",
    files: ["/demo_samples/UseCase3_Change_Before.jpg", "/demo_samples/UseCase3_Change_After.jpg"],
    timestamps: "2023-01-15, 2024-01-15",
    task: "Change Detection"
  },
  optical_sar: {
    title: "Optical + SAR Cross-Modal Fusion",
    query: "Perform joint Optical and SAR analysis to classify land cover.",
    files: ["/demo_samples/UseCase4_Fusion_Optical.jpg", "/demo_samples/UseCase4_Fusion_SAR.jpg"],
    timestamps: "2023-08-10, 2023-08-10",
    task: "Cross-Modal Fusion"
  },
  captioning: {
    title: "Autonomous Scene Captioning",
    query: "Describe the dominant terrain and infrastructure.",
    files: ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"],
    timestamps: "",
    task: "Scene Captioning"
  }
};

// 1-Click Demo Loader
window.loadDemo = async function(demoKey) {
  const preset = DEMO_PRESETS[demoKey];
  if (!preset) return;

  // Highlight active button
  document.querySelectorAll(".demo-btn").forEach(btn => btn.classList.remove("active-demo"));
  const clickedBtn = Array.from(document.querySelectorAll(".demo-btn")).find(b => b.getAttribute("onclick")?.includes(demoKey));
  if (clickedBtn) clickedBtn.classList.add("active-demo");

  queryInput.value = preset.query;
  timestampsInput.value = preset.timestamps || "";
  filePreviews.innerHTML = `<span style="font-size: 11px; color: var(--accent-cyan); font-family: var(--font-mono);">Loading mission assets...</span>`;

  try {
    const fetchedFiles = [];
    for (const url of preset.files) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Could not fetch ${url}`);
      const blob = await resp.blob();
      const fname = url.split("/").pop();
      fetchedFiles.push(new File([blob], fname, { type: "image/jpeg" }));
    }

    handleFiles(fetchedFiles);

    // Auto-trigger analysis for seamless showcase
    setTimeout(() => {
      executeAnalysis();
    }, 150);
  } catch (err) {
    filePreviews.innerHTML = `<span style="font-size: 11px; color: var(--accent-red);">Failed to load demo files: ${err.message}</span>`;
  }
};

// Quick Query Setter
window.setQuery = function(text) {
  queryInput.value = text;
};

// File Selection & Drag-and-Drop Handlers
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--accent-cyan)";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "var(--border)";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--border)";
  handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", (e) => {
  handleFiles(e.target.files);
});

function handleFiles(files) {
  selectedFiles = Array.from(files).slice(0, 2);
  filePreviews.innerHTML = "";

  if (selectedFiles.length === 0) return;

  selectedFiles.forEach((file, idx) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const thumbWrap = document.createElement("div");
      thumbWrap.style.display = "flex";
      thumbWrap.style.flexDirection = "column";
      thumbWrap.style.alignItems = "center";
      thumbWrap.style.gap = "4px";

      const img = document.createElement("img");
      img.src = e.target.result;
      img.className = "preview-thumb";
      img.title = `Image ${idx + 1}: ${file.name}`;

      const lbl = document.createElement("span");
      lbl.style.fontSize = "9px";
      lbl.style.color = "var(--accent-cyan)";
      lbl.style.fontFamily = "var(--font-mono)";
      lbl.textContent = (selectedFiles.length === 2) ? (idx === 0 ? "T1 / Optical" : "T2 / SAR") : "Scene";

      thumbWrap.appendChild(img);
      thumbWrap.appendChild(lbl);
      filePreviews.appendChild(thumbWrap);
    };
    reader.readAsDataURL(file);
  });

  // Show timestamps input if exactly 2 images uploaded (bi-temporal)
  if (selectedFiles.length === 2) {
    temporalGroup.style.display = "flex";
    if (!timestampsInput.value) {
      timestampsInput.value = "2023-01-15, 2024-01-15";
    }
  } else {
    temporalGroup.style.display = "none";
  }
}

// Split Comparison Slider Logic
compareSlider.addEventListener("input", (e) => {
  const val = e.target.value;
  afterSliderWrapper.style.width = `${val}%`;
  sliderLine.style.left = `${val}%`;
});

// Execute Analysis Trigger
analyzeBtn.addEventListener("click", () => {
  executeAnalysis();
});

async function executeAnalysis() {
  const query = queryInput.value.trim();
  if (!query) {
    alert("Please enter a query or instruction.");
    return;
  }
  if (selectedFiles.length === 0) {
    alert("Please upload at least 1 satellite image or click a 1-Click Demo.");
    return;
  }

  // Set Loading State
  analyzeBtn.disabled = true;
  btnText.textContent = "Orchestrating Specialists...";
  btnSpinner.style.display = "inline-block";

  const formData = new FormData();
  formData.append("query", query);
  selectedFiles.forEach((file) => {
    formData.append("images", file);
  });

  if (timestampsInput.value.trim()) {
    formData.append("timestamps", timestampsInput.value.trim());
  }

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Analysis request failed.");
    }

    const data = await response.json();
    latestAnalysisResult = data;
    renderResults(data);
  } catch (error) {
    alert(`Error: ${error.message}`);
  } finally {
    analyzeBtn.disabled = false;
    btnText.textContent = "Execute SatQuery Orchestrator";
    btnSpinner.style.display = "none";
  }
}

// Trace Step Icons Map
const STEP_ICONS = {
  1: "🛡️",
  2: "🧭",
  3: "⚡",
  4: "📊",
  5: "🏁"
};

// Render Results & Visual Evidence
function renderResults(result) {
  // 1. Telemetry HUD Bar
  hudConf.textContent = `${(result.confidence_score * 100).toFixed(1)}%`;
  hudTime.textContent = `${result.processing_time_ms} ms`;
  
  const plannedTask = result.execution_trace.find(s => s.step_name.includes("Task Classification"))?.parameters?.primary_task || "vqa";
  hudTask.textContent = plannedTask.replace(/_/g, " ").toUpperCase();
  
  const evType = result.evidence?.evidence_type || "none";
  hudEvidence.textContent = evType === "none" ? "Semantic Text" : evType.toUpperCase();

  // 2. Executive Intelligence Text Answer
  answerContainer.innerHTML = `
    <div class="answer-header">Verified Intelligence Output:</div>
    <p style="color: var(--text-main); line-height: 1.6;">${result.answer}</p>
  `;

  // 3. Confidence Badge
  confBadge.textContent = `${result.confidence_level} (${(result.confidence_score * 100).toFixed(1)}%)`;
  confBadge.className = `confidence-badge ${result.confidence_level}`;

  // 4. Observable Execution Trace
  traceTimeline.innerHTML = "";
  result.execution_trace.forEach((step) => {
    const icon = STEP_ICONS[step.step_number] || "🔹";
    const stepCard = document.createElement("div");
    stepCard.className = "timeline-step";
    stepCard.innerHTML = `
      <div class="step-meta">
        <span class="step-name">${icon} Step ${step.step_number}: ${step.step_name}</span>
        <span class="step-tool">${step.tool_used || "SatQuery-Core"}</span>
      </div>
      <div class="step-action">${step.action_taken}</div>
      <div class="step-summary">Output: ${step.output_summary}</div>
    `;
    traceTimeline.appendChild(stepCard);
  });

  // 5. Visual Evidence Viewer
  emptyViewer.style.display = "none";
  bboxOverlay.innerHTML = "";
  toggleMaskBtn.style.display = "none";

  const evidence = result.evidence;
  evidenceTag.textContent = evType.replace(/_/g, " ").toUpperCase();

  if (selectedFiles.length === 2) {
    // Bi-Temporal or Cross-Modal Pair Mode
    singleWrapper.style.display = "none";
    comparisonWrapper.style.display = "block";

    const r1 = new FileReader();
    r1.onload = (e) => { imgBefore.src = e.target.result; };
    r1.readAsDataURL(selectedFiles[0]);

    const r2 = new FileReader();
    r2.onload = (e) => { 
      imgAfter.src = e.target.result;
      imgAfter.dataset.originalSrc = e.target.result;
    };
    r2.readAsDataURL(selectedFiles[1]);

    if (evidence && evidence.evidence_type === "change_mask" && evidence.mask_path) {
      toggleMaskBtn.style.display = "inline-block";
      toggleMaskBtn.textContent = "🔥 Show Glowing Heatmap";
      imgAfter.dataset.isMask = "false";
      tagAfter.textContent = "T2: After";

      toggleMaskBtn.onclick = () => {
        if (imgAfter.dataset.isMask === "true") {
          imgAfter.src = imgAfter.dataset.originalSrc;
          imgAfter.dataset.isMask = "false";
          toggleMaskBtn.textContent = "🔥 Show Glowing Heatmap";
          tagAfter.textContent = "T2: After";
        } else {
          // Normalize path for web
          const maskUrl = evidence.mask_path.startsWith("http") ? evidence.mask_path : `${API_BASE}${evidence.mask_path.startsWith("/") ? "" : "/"}${evidence.mask_path}`;
          imgAfter.src = maskUrl;
          imgAfter.dataset.isMask = "true";
          toggleMaskBtn.textContent = "🖼️ Show Post-Event Photo";
          tagAfter.textContent = "T2: Heatmap Overlay";
        }
      };
    }
  } else {
    // Single Image with Bounding Boxes
    comparisonWrapper.style.display = "none";
    singleWrapper.style.display = "flex";

    const reader = new FileReader();
    reader.onload = (e) => {
      mainDisplayImg.src = e.target.result;
      
      // Draw Bounding Boxes if present
      if (evidence && evidence.bounding_boxes && evidence.bounding_boxes.length > 0) {
        evidence.bounding_boxes.forEach((box, i) => {
          const [ymin, xmin, ymax, xmax] = box;
          const boxEl = document.createElement("div");
          boxEl.className = "bbox-box";
          boxEl.style.top = `${ymin * 100}%`;
          boxEl.style.left = `${xmin * 100}%`;
          boxEl.style.height = `${(ymax - ymin) * 100}%`;
          boxEl.style.width = `${(xmax - xmin) * 100}%`;

          const labelText = (evidence.labels && evidence.labels[i]) ? evidence.labels[i] : `Target ${i+1}`;
          const labelEl = document.createElement("div");
          labelEl.className = "bbox-label";
          labelEl.textContent = labelText;
          boxEl.appendChild(labelEl);

          bboxOverlay.appendChild(boxEl);
        });
      }
    };
    reader.readAsDataURL(selectedFiles[0]);
  }

  // Enable Report Download
  downloadReportBtn.disabled = false;
}

// Download Analytical Report
downloadReportBtn.addEventListener("click", () => {
  if (!latestAnalysisResult) return;

  const blob = new Blob([JSON.stringify(latestAnalysisResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `satquery_certified_mission_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

