const queryForm = document.getElementById("queryForm");
const queryInput = document.getElementById("queryInput");

const attachButton = document.getElementById("attachButton");
const fileInput = document.getElementById("fileInput");

const fileList = document.getElementById("fileList");

const thinkButton = document.getElementById("thinkButton");
const micButton = document.getElementById("micButton");

let selectedFiles = [];


// ================================
// SATELLITE MAP APERTURE & THIN LIGHT RING
// ================================

const magnifierCursor = document.getElementById("magnifierCursor");
const satelliteMapBg = document.getElementById("satelliteMapBg");
const queryFormElement = document.getElementById("queryForm");

let targetX = window.innerWidth / 2;
let targetY = window.innerHeight / 2;
let currentX = targetX;
let currentY = targetY;
let isCursorVisible = false;
let isCursorOnSearchBar = false;

const BASE_RADIUS = 115;
const FOCUS_RADIUS = 135;
let targetRadius = BASE_RADIUS;
let currentRadius = BASE_RADIUS;

function isOverSearchBar(e) {
    if (!queryFormElement) return false;
    const rect = queryFormElement.getBoundingClientRect();
    // Add 4px tolerance around search form
    return (
        e.clientX >= rect.left - 4 &&
        e.clientX <= rect.right + 4 &&
        e.clientY >= rect.top - 4 &&
        e.clientY <= rect.bottom + 4
    );
}

document.addEventListener("mousemove", (e) => {
    targetX = e.clientX;
    targetY = e.clientY;

    if (isOverSearchBar(e)) {
        isCursorOnSearchBar = true;
        if (magnifierCursor) magnifierCursor.style.opacity = "0";
        if (satelliteMapBg) satelliteMapBg.style.opacity = "0";
        return;
    } else {
        if (isCursorOnSearchBar) {
            isCursorOnSearchBar = false;
        }
    }

    if (!isCursorVisible) {
        isCursorVisible = true;
    }
    if (magnifierCursor) magnifierCursor.style.opacity = "1";
    if (satelliteMapBg) satelliteMapBg.style.opacity = "1";
});

document.addEventListener("mouseleave", () => {
    isCursorVisible = false;
    if (magnifierCursor) magnifierCursor.style.opacity = "0";
    if (satelliteMapBg) satelliteMapBg.style.opacity = "0";
});

document.addEventListener("mouseenter", (e) => {
    if (!isOverSearchBar(e)) {
        isCursorVisible = true;
        if (magnifierCursor) magnifierCursor.style.opacity = "1";
        if (satelliteMapBg) satelliteMapBg.style.opacity = "1";
    }
});

// Search bar boundary listeners
if (queryFormElement) {
    queryFormElement.addEventListener("mouseenter", () => {
        isCursorOnSearchBar = true;
        if (magnifierCursor) magnifierCursor.style.opacity = "0";
        if (satelliteMapBg) satelliteMapBg.style.opacity = "0";
    });
    queryFormElement.addEventListener("mouseleave", (e) => {
        if (!isOverSearchBar(e)) {
            isCursorOnSearchBar = false;
            if (magnifierCursor) magnifierCursor.style.opacity = "1";
            if (satelliteMapBg) satelliteMapBg.style.opacity = "1";
        }
    });
}

function animateMagnifier() {
    currentX += (targetX - currentX) * 0.38;
    currentY += (targetY - currentY) * 0.38;
    currentRadius += (targetRadius - currentRadius) * 0.22;

    const roundX = Math.round(currentX);
    const roundY = Math.round(currentY);
    const roundR = Math.round(currentRadius);

    if (magnifierCursor) {
        magnifierCursor.style.transform = `translate3d(${roundX}px, ${roundY}px, 0)`;
    }

    if (satelliteMapBg) {
        const maskGrad = `radial-gradient(circle ${roundR}px at ${roundX}px ${roundY}px, black 0%, black 90%, transparent 100%)`;
        satelliteMapBg.style.webkitMaskImage = maskGrad;
        satelliteMapBg.style.maskImage = maskGrad;
    }

    requestAnimationFrame(animateMagnifier);
}
requestAnimationFrame(animateMagnifier);

// Click flash & tactile ring bounce
document.addEventListener("mousedown", (e) => {
    if (!isCursorOnSearchBar && magnifierCursor) {
        magnifierCursor.classList.add("magnifier-click");
        targetRadius = BASE_RADIUS * 0.95;
    }
});

document.addEventListener("mouseup", () => {
    if (magnifierCursor) magnifierCursor.classList.remove("magnifier-click");
    targetRadius = BASE_RADIUS;
});

// Subtle expansion when hovering preset chips
function setupMagnifierFocusListeners() {
    document.querySelectorAll(".preset-chip").forEach(el => {
        el.addEventListener("mouseenter", () => {
            targetRadius = FOCUS_RADIUS;
        });
        el.addEventListener("mouseleave", () => {
            targetRadius = BASE_RADIUS;
        });
    });
}
setupMagnifierFocusListeners();

const inputContainer = document.querySelector(".input-container");



// ================================
// ATTACH BUTTON & DRAG/DROP
// ================================

attachButton.addEventListener("click", () => {
    fileInput.click();
});

// Drag and drop onto input container
if (inputContainer) {
    inputContainer.addEventListener("dragover", (e) => {
        e.preventDefault();
        inputContainer.classList.add("drag-over");
    });

    inputContainer.addEventListener("dragleave", (e) => {
        if (!inputContainer.contains(e.relatedTarget)) {
            inputContainer.classList.remove("drag-over");
        }
    });

    inputContainer.addEventListener("drop", (e) => {
        e.preventDefault();
        inputContainer.classList.remove("drag-over");
        if (e.dataTransfer && e.dataTransfer.files.length > 0) {
            handleIncomingFiles(Array.from(e.dataTransfer.files));
        }
    });
}


// ================================
// FILE SELECTION & PROCESSING
// ================================

fileInput.addEventListener("change", () => {
    const newFiles = Array.from(fileInput.files);
    fileInput.value = ""; // Reset to allow re-selecting same file
    handleIncomingFiles(newFiles);
});

function handleIncomingFiles(newFiles) {
    const validFiles = newFiles.filter(f =>
        f.type.startsWith("image/") || f.name.match(/\.(tif|tiff|jpg|jpeg|png|webp)$/i)
    );

    if (validFiles.length === 0) {
        alert("Please upload a valid satellite image (JPEG, PNG, TIFF, WEBP).");
        return;
    }

    if (selectedFiles.length + validFiles.length > 2) {
        alert("You can upload a maximum of 2 images for analysis & comparison.");
        return;
    }

    validFiles.forEach(file => {
        // Read image immediately for large instant preview & caching
        const reader = new FileReader();
        reader.onload = (e) => {
            file._dataUrl = e.target.result;
            displayFiles();
        };
        reader.readAsDataURL(file);
        selectedFiles.push(file);
    });

    displayFiles();
}

function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}


// ================================
// LARGE PHOTO PREVIEW RENDERER
// ================================

function displayFiles() {
    fileList.innerHTML = "";

    if (selectedFiles.length === 0) {
        fileList.classList.remove("multi-files");
        return;
    }

    if (selectedFiles.length > 1) {
        fileList.classList.add("multi-files");
    } else {
        fileList.classList.remove("multi-files");
    }

    selectedFiles.forEach((file, index) => {
        const card = document.createElement("div");
        card.className = "file-preview-card";

        const badgeLabel = selectedFiles.length > 1
            ? (index === 0 ? "IMAGE #1 • BEFORE" : "IMAGE #2 • AFTER")
            : "SATELLITE IMAGERY";

        card.innerHTML = `
            <div class="file-preview-media">
                <img src="${file._dataUrl || ''}" alt="${file.name}" class="file-preview-img" title="Click to view full size">
                <div class="file-preview-gradient"></div>

                <div class="file-preview-top-bar">
                    <span class="file-preview-badge">
                        <span class="status-dot"></span>
                        ${badgeLabel}
                    </span>
                    <div class="file-preview-actions">
                        <button type="button" class="preview-action-btn view-full-btn" data-index="${index}" title="View full size">
                            ⛶
                        </button>
                        <button type="button" class="preview-action-btn remove-file-btn" data-index="${index}" title="Remove image">
                            ✕
                        </button>
                    </div>
                </div>

                <div class="file-preview-bottom-bar">
                    <div class="file-preview-details">
                        <span class="file-preview-filename" title="${file.name}">${file.name}</span>
                        <span class="file-preview-size">${formatBytes(file.size)}</span>
                    </div>
                    <span class="file-preview-status">✓ Ready for Analysis</span>
                </div>
            </div>
        `;

        // Click on img or full-size button opens lightbox modal
        const imgEl = card.querySelector(".file-preview-img");
        const fullBtn = card.querySelector(".view-full-btn");

        if (file.name.match(/\.(tif|tiff)$/i)) {
            imgEl.onerror = () => {
                imgEl.style.display = "none";
                const fallbackBadge = document.createElement("div");
                fallbackBadge.style.cssText = "display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#38bdf8;font-size:12px;font-family:monospace;letter-spacing:1px;text-align:center;padding:15px;background:rgba(15,23,42,0.92);";
                fallbackBadge.innerHTML = `<span style="font-size:30px;margin-bottom:6px;">🛰️</span><strong style="color:#f8fafc;">GEOTIFF SATELLITE IMAGE</strong><span style="font-size:10px;color:#94a3b8;margin-top:4px;">Multi-Band Remote Sensing Array</span>`;
                imgEl.parentNode.insertBefore(fallbackBadge, imgEl);
            };
        }

        const openModal = () => {
            if (file._dataUrl && !file.name.match(/\.(tif|tiff)$/i)) {
                openLightbox(file._dataUrl, file.name);
            }
        };
        imgEl.addEventListener("click", openModal);
        fullBtn.addEventListener("click", openModal);

        // Remove button
        const removeBtn = card.querySelector(".remove-file-btn");
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            selectedFiles.splice(index, 1);
            displayFiles();
        });

        fileList.appendChild(card);
    });
}


// ================================
// IMAGE LIGHTBOX MODAL (FULL SIZE PREVIEW)
// ================================

const imageLightbox = document.getElementById("imageLightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxCaption = document.getElementById("lightboxCaption");
const lightboxClose = document.getElementById("lightboxClose");

function openLightbox(url, caption) {
    if (!imageLightbox) return;
    lightboxImg.src = url;
    lightboxCaption.textContent = caption || "Satellite Image Preview";
    imageLightbox.style.display = "flex";
}

function closeLightbox() {
    if (!imageLightbox) return;
    imageLightbox.style.display = "none";
    lightboxImg.src = "";
}

if (lightboxClose) {
    lightboxClose.addEventListener("click", closeLightbox);
}

if (imageLightbox) {
    imageLightbox.addEventListener("click", (e) => {
        if (e.target === imageLightbox) {
            closeLightbox();
        }
    });
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && imageLightbox && imageLightbox.style.display === "flex") {
        closeLightbox();
    }
});


// ================================
// THINK BUTTON (IF PRESENT)
// ================================

if (thinkButton) {
    thinkButton.addEventListener("click", () => {
        thinkButton.classList.toggle("active");
    });
}


// ================================
// MICROPHONE
// ================================

micButton.addEventListener("click", () => {

    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    if (!SpeechRecognition) {

        alert(
            "Speech recognition is not supported in this browser."
        );

        return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-IN";

    recognition.start();

    micButton.style.color = "#ff5265";

    recognition.onresult = (event) => {

        const transcript =
            event.results[0][0].transcript;

        queryInput.value = transcript;
    };

    recognition.onend = () => {

        micButton.style.color = "";
    };

});


// ================================
// PRESET BUTTONS
// ================================

document.querySelectorAll(".preset-chip").forEach(button => {

    button.addEventListener("click", () => {

        const demo = button.dataset.demo;

        launchDemo(demo);

    });

});


// ================================
// SMOOTH PAGE NAVIGATION
// Slides the landing page up and out before redirecting
// ================================

function navigateTo(url) {
    document.body.classList.add("page-slide-out");
    setTimeout(() => {
        window.location.href = url;
    }, 400);
}


// ================================
// DEMO LAUNCHER
// ================================

const DEMO_PRESETS = {
    grounding: {
        query: "Where are the water bodies located in this scene?",
        images: ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"]
    },
    change: {
        query: "What changed between these observation dates?",
        images: ["/demo_samples/UseCase3_Change_Before.jpg", "/demo_samples/UseCase3_Change_After.jpg"]
    },
    optical_sar: {
        query: "Perform joint Optical and SAR analysis to classify land cover.",
        images: ["/demo_samples/UseCase4_Fusion_Optical.jpg", "/demo_samples/UseCase4_Fusion_SAR.jpg"]
    },
    captioning: {
        query: "Describe the dominant terrain and infrastructure.",
        images: ["/demo_samples/UseCase1_VQA_and_Grounding.jpg"]
    }
};

function launchDemo(demo) {
    const p = DEMO_PRESETS[demo] || DEMO_PRESETS.captioning;
    const query = p.query;

    sessionStorage.setItem("satquery_query", query);
    sessionStorage.setItem("satquery_image", p.images[0]);
    if (p.images[1]) {
        sessionStorage.setItem("satquery_image_2", p.images[1]);
    } else {
        sessionStorage.removeItem("satquery_image_2");
    }

    navigateTo(
        `output.html?query=${encodeURIComponent(query)}&demo=${demo}`
    );
}


// ================================
// FORM SUBMISSION
// ================================

queryForm.addEventListener("submit", (event) => {

    event.preventDefault();

    const query = queryInput.value.trim();

    if (!query) {
        alert("Please enter a question.");
        return;
    }

    sessionStorage.setItem(
        "satquery_query",
        query
    );

    if (selectedFiles.length > 0) {

        const proceedWithSubmission = () => {
            sessionStorage.setItem("satquery_image", selectedFiles[0]._dataUrl);
            sessionStorage.setItem("satquery_filename", selectedFiles[0].name || "satellite.tif");

            if (selectedFiles.length > 1 && selectedFiles[1]._dataUrl) {
                sessionStorage.setItem("satquery_image_2", selectedFiles[1]._dataUrl);
                sessionStorage.setItem("satquery_filename_2", selectedFiles[1].name || "satellite_2.tif");
                navigateTo(`output.html?query=${encodeURIComponent(query)}&demo=change`);
            } else {
                sessionStorage.removeItem("satquery_image_2");
                sessionStorage.removeItem("satquery_filename_2");
                navigateTo(`output.html?query=${encodeURIComponent(query)}`);
            }
        };

        // If dataUrl already cached, navigate immediately
        if (selectedFiles[0]._dataUrl) {
            proceedWithSubmission();
        } else {
            const reader = new FileReader();
            reader.onload = function(e) {
                selectedFiles[0]._dataUrl = e.target.result;
                proceedWithSubmission();
            };
            reader.readAsDataURL(selectedFiles[0]);
        }

    } else {
        sessionStorage.removeItem("satquery_image");
        sessionStorage.removeItem("satquery_image_2");
        sessionStorage.removeItem("satquery_filename");
        sessionStorage.removeItem("satquery_filename_2");
        navigateTo(`output.html?query=${encodeURIComponent(query)}`);
    }

});