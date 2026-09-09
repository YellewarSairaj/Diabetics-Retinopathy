/**
 * RetinaX AI — Enterprise Frontend Application Logic
 * ==================================================
 * Features:
 *  - Theme Engine: Dark Mode (Default) / Light Mode with localStorage persistence
 *  - SPA Router: Navigation between Home, Dashboard, Prediction, Grad-CAM, Reports, About, Contact
 *  - REST API Engine: Integrates with FastAPI endpoints (/health, /metrics, /api/v1/explain, /api/v1/report/pdf, /history)
 *  - State Synchronizer: Auto-populates Grad-CAM & Reports views on prediction run
 */

const API_BASE_URL = "http://localhost:8000";

// DR Severity Color Definitions
const SEVERITY_COLORS = [
    "#22C55E", // 0: No DR (Green)
    "#3B82F6", // 1: Mild (Blue)
    "#EAB308", // 2: Moderate (Amber)
    "#F97316", // 3: Severe (Orange)
    "#EF4444"  // 4: Proliferative (Red)
];

const CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

// Global State
let selectedFile = null;
let currentExplainData = null;
let originalBase64 = null;

/* ==========================================================================
   1. THEME SWITCHER & INITIALIZATION
   ========================================================================== */

function initTheme() {
    const savedTheme = localStorage.getItem("retinax-theme") || "dark";
    applyTheme(savedTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("retinax-theme", theme);
    
    const themeIcon = document.getElementById("theme-icon");
    if (themeIcon) {
        themeIcon.setAttribute("data-lucide", theme === "dark" ? "sun" : "moon");
        if (window.lucide) lucide.createIcons();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    if (window.lucide) lucide.createIcons();
    
    // Check Backend Server Health & Fetch Metrics
    checkServerHealth();
    fetchMetrics();
    fetchAuditHistory();
    setupEventListeners();
});

/* ==========================================================================
   2. SINGLE PAGE APPLICATION (SPA) ROUTER
   ========================================================================== */

function navigateToView(viewName) {
    const targetSection = document.getElementById(`view-${viewName}`);
    if (!targetSection) return;

    // Hide all view sections
    document.querySelectorAll(".view-section").forEach(sec => sec.classList.remove("active"));
    
    // Show target section
    targetSection.classList.add("active");

    // Update nav links active state
    document.querySelectorAll(".nav-links .nav-item").forEach(item => {
        if (item.getAttribute("data-view") === viewName) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    // Close mobile hamburger menu if open
    const navMenu = document.getElementById("nav-menu");
    if (navMenu) navMenu.classList.remove("mobile-open");

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function setupEventListeners() {
    // Nav links click routing
    document.querySelectorAll("[data-view]").forEach(elem => {
        elem.addEventListener("click", (e) => {
            e.preventDefault();
            const view = elem.getAttribute("data-view");
            if (view) navigateToView(view);
        });
    });

    // Theme Toggle Button
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme");
            applyTheme(currentTheme === "dark" ? "light" : "dark");
        });
    }

    // Hamburger Mobile Menu Toggle
    const hamburgerBtn = document.getElementById("hamburger-btn");
    const navMenu = document.getElementById("nav-menu");
    if (hamburgerBtn && navMenu) {
        hamburgerBtn.addEventListener("click", () => {
            navMenu.classList.toggle("mobile-open");
        });
    }

    // Refresh Audit History Button
    const refreshBtn = document.getElementById("refresh-history-btn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", fetchAuditHistory);
    }

    // Dropzone & File Input Handlers
    setupDropzone();

    // Analyze Button
    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", runAnalysis);
    }

    // Download PDF Button
    const downloadPdfBtn = document.getElementById("download-pdf-btn");
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener("click", downloadPdfReport);
    }
}

/* ==========================================================================
   3. REST API INTEGRATION
   ========================================================================== */

async function checkServerHealth() {
    const statusText = document.getElementById("system-status-text");
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            if (statusText) statusText.textContent = `Online: ${data.model_architecture.split(' ')[0]}`;
        }
    } catch (e) {
        if (statusText) statusText.textContent = "Backend Offline";
    }
}

async function fetchMetrics() {
    try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (res.ok) {
            const data = await res.json();
            if (data.accuracy) document.getElementById("stat-accuracy").textContent = `${(data.accuracy * 100).toFixed(1)}%`;
            if (data.roc_auc) document.getElementById("stat-auc").textContent = `${(data.roc_auc * 100).toFixed(1)}%`;
            if (data.f1_score) document.getElementById("stat-f1").textContent = `${(data.f1_score * 100).toFixed(1)}%`;
        }
    } catch (e) {
        console.warn("Could not fetch live metrics:", e);
    }
}

async function fetchAuditHistory() {
    const tbody = document.getElementById("audit-table-body");
    if (!tbody) return;

    try {
        const res = await fetch(`${API_BASE_URL}/api/v1/predictions/history?limit=10`);
        if (res.ok) {
            const data = await res.json();
            if (data.records && data.records.length > 0) {
                tbody.innerHTML = "";
                data.records.forEach(r => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>#${r.id}</td>
                        <td><strong>${r.patient_id}</strong></td>
                        <td>${r.filename}</td>
                        <td><span class="badge badge-${r.predicted_index || 0}">${r.predicted_class}</span></td>
                        <td>${(r.confidence * 100).toFixed(1)}%</td>
                        <td style="font-size:0.8rem; color:var(--text-secondary);">${r.recommendation}</td>
                        <td>${r.created_at}</td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-secondary);">No clinical audit records found.</td></tr>`;
            }
        }
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-secondary);">Unable to load audit history. Start FastAPI server.</td></tr>`;
    }
}

/* ==========================================================================
   4. DROPZONE & FILE INPUT HANDLERS
   ========================================================================== */

function setupDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeImgBtn = document.getElementById("remove-img-btn");
    const analyzeBtn = document.getElementById("analyze-btn");

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener("click", (e) => {
        if (e.target !== removeImgBtn && !removeImgBtn?.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    removeImgBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        originalBase64 = null;
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        if (analyzeBtn) analyzeBtn.disabled = true;
    });

    // Sample Thumbs click
    document.querySelectorAll(".sample-thumb").forEach(thumb => {
        thumb.addEventListener("click", async () => {
            try {
                const response = await fetch(thumb.src);
                const blob = await response.blob();
                const file = new File([blob], "sample_fundus.png", { type: "image/png" });
                handleFileSelect(file);
            } catch (err) {
                console.warn("Failed to load sample image:", err);
            }
        });
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid fundus image file (.png, .jpg, .jpeg).");
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        originalBase64 = e.target.result;
        document.getElementById("image-preview").src = originalBase64;
        document.getElementById("dropzone-prompt").classList.add("hidden");
        document.getElementById("preview-container").classList.remove("hidden");
        const analyzeBtn = document.getElementById("analyze-btn");
        if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

/* ==========================================================================
   5. INFERENCE & STATE SYNCHRONIZER
   ========================================================================== */

async function runAnalysis() {
    if (!selectedFile) return;

    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `<i data-lucide="loader"></i> Executing Inference Pipeline...`;
        if (window.lucide) lucide.createIcons();
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/explain`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Inference request failed.");

        currentExplainData = await response.json();
        
        // Sync results across UI views
        renderPredictionResults(currentExplainData);
        syncGradCamView(currentExplainData);
        syncReportView(currentExplainData);
        
        // Refresh Audit History
        fetchAuditHistory();
    } catch (err) {
        alert(`Error during analysis: ${err.message}`);
    } finally {
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `<i data-lucide="cpu"></i> Run Deep Learning & XAI Diagnosis`;
            if (window.lucide) lucide.createIcons();
        }
    }
}

function renderPredictionResults(data) {
    const pred = data.prediction;

    document.getElementById("prediction-idle-state")?.classList.add("hidden");
    document.getElementById("prediction-results-state")?.classList.remove("hidden");

    const resClass = document.getElementById("res-pred-class");
    const resRec = document.getElementById("res-recommendation");
    const resConf = document.getElementById("res-confidence");
    const banner = document.getElementById("result-card-banner");

    if (resClass) resClass.textContent = pred.predicted_class;
    if (resRec) resRec.textContent = pred.clinical_recommendation;
    if (resConf) resConf.textContent = `${(pred.confidence * 100).toFixed(1)}%`;

    const accentColor = SEVERITY_COLORS[pred.predicted_index] || "#2563EB";
    if (banner) banner.style.borderLeftColor = accentColor;
    if (resClass) resClass.style.color = accentColor;

    // Render Probability Distribution Bars
    const container = document.getElementById("prob-bars-container");
    if (container) {
        container.innerHTML = "";
        pred.class_probabilities.forEach(item => {
            const percentage = (item.probability * 100).toFixed(1);
            const color = SEVERITY_COLORS[item.class_index];
            const isTarget = item.class_index === pred.predicted_index;

            const row = document.createElement("div");
            row.className = "prob-item";
            row.innerHTML = `
                <div class="prob-item-info">
                    <span class="${isTarget ? 'font-bold' : ''}" style="${isTarget ? 'color:'+color : ''}">
                        ${item.class_name} ${isTarget ? '★' : ''}
                    </span>
                    <span>${percentage}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:${percentage}%; background-color:${color}"></div>
                </div>
            `;
            container.appendChild(row);
        });
    }
}

function syncGradCamView(data) {
    const origImg = document.getElementById("xai-original-img");
    const heatImg = document.getElementById("xai-heatmap-img");
    const overlayImg = document.getElementById("xai-overlay-img");

    if (origImg) origImg.src = originalBase64;
    if (heatImg) heatImg.src = data.gradcam_heatmap_base64;
    if (overlayImg) overlayImg.src = data.clinical_overlay_base64;
}

function syncReportView(data) {
    const pred = data.prediction;
    
    document.getElementById("rpt-patient-id").textContent = `PAT-${selectedFile?.name.substring(0, 8) || '8823901'}`;
    document.getElementById("rpt-date").textContent = new Date().toISOString().split('T')[0];
    
    const rptDiag = document.getElementById("rpt-diagnosis");
    if (rptDiag) {
        rptDiag.textContent = pred.predicted_class;
        rptDiag.style.color = SEVERITY_COLORS[pred.predicted_index] || "#2563EB";
    }

    document.getElementById("rpt-recommendation").textContent = pred.clinical_recommendation;
    
    const rptOverlay = document.getElementById("rpt-overlay-img");
    if (rptOverlay) rptOverlay.src = data.clinical_overlay_base64;
}

/* ==========================================================================
   6. PDF REPORT DOWNLOAD
   ========================================================================== */

async function downloadPdfReport() {
    if (!selectedFile) {
        alert("Please upload and analyze a fundus image first.");
        return;
    }

    const btn = document.getElementById("download-pdf-btn");
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader"></i> Generating Medical PDF...`;
        if (window.lucide) lucide.createIcons();
    }

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_BASE_URL}/api/v1/report/pdf`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Failed to generate PDF report.");

        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = `RetinaX_Medical_Report_${selectedFile.name}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    } catch (e) {
        alert(`PDF Export Error: ${e.message}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i data-lucide="file-text"></i> Download Medical PDF Report`;
            if (window.lucide) lucide.createIcons();
        }
    }
}
