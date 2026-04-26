/* ──────────────────────────────────────────────────────────────
   NeuroScan AI  –  script.js
   Handles: file selection, drag-and-drop, API call, result render
────────────────────────────────────────────────────────────── */

// ── CONFIG  ─────────────────────────────────────────────────────
// 🔴 Replace this URL with your actual Render backend URL after deploy
const API_URL = "https://YOUR-APP-NAME.onrender.com/predict";

// ── DOM REFS ─────────────────────────────────────────────────────
const fileInput      = document.getElementById("fileInput");
const dropZone       = document.getElementById("dropZone");
const dropInner      = document.getElementById("dropInner");
const preview        = document.getElementById("preview");
const analyzeBtn     = document.getElementById("analyzeBtn");
const btnText        = analyzeBtn.querySelector(".btn-text");
const btnLoader      = document.getElementById("btnLoader");
const uploadCard     = document.getElementById("uploadCard");
const resultCard     = document.getElementById("resultCard");
const resetBtn       = document.getElementById("resetBtn");
const errorBanner    = document.getElementById("errorBanner");

// Result fields
const predBanner     = document.getElementById("predBanner");
const predClass      = document.getElementById("predClass");
const predConf       = document.getElementById("predConf");
const severityBadge  = document.getElementById("severityBadge");
const reportExpl     = document.getElementById("reportExplanation");
const reportPrec     = document.getElementById("reportPrecautions");
const reportLife     = document.getElementById("reportLifestyle");
const reportDisc     = document.getElementById("reportDisclaimer");

// Severity → border colour map
const SEVERITY_COLORS = {
  "None":      "#22c55e",
  "Very Mild": "#eab308",
  "Mild":      "#f97316",
  "Moderate":  "#ef4444",
};

// ── FILE SELECTION ────────────────────────────────────────────────
let selectedFile = null;

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── DRAG & DROP ───────────────────────────────────────────────────
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

["dragleave", "dragend"].forEach((ev) =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove("dragover"))
);

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
  else showError("Please drop a valid image file (JPG or PNG).");
});

// ── HANDLE FILE ───────────────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;
  hideError();

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.remove("hidden");
    dropInner.classList.add("hidden");
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
}

// ── ANALYSE ───────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(
      `❌ ${err.message}. ` +
      `Make sure the backend is running and API_URL is set correctly in script.js.`
    );
  } finally {
    setLoading(false);
  }
});

// ── RENDER RESULT ─────────────────────────────────────────────────
function renderResult(data) {
  // Prediction banner
  predClass.textContent = data.predicted_class;
  predConf.textContent  = `${data.confidence}%`;
  severityBadge.textContent = `Severity: ${data.severity}`;

  const color = SEVERITY_COLORS[data.severity] || "#0ea5e9";
  predBanner.style.borderLeftColor = color;
  severityBadge.style.background   = `${color}22`;
  severityBadge.style.color        = color;

  // Report body
  reportExpl.textContent = data.explanation;

  populateList(reportPrec, data.precautions);
  populateList(reportLife, data.lifestyle);

  reportDisc.textContent = data.disclaimer;

  // Show result, hide upload
  uploadCard.classList.add("hidden");
  resultCard.classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function populateList(ulEl, items) {
  ulEl.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    ulEl.appendChild(li);
  });
}

// ── RESET ─────────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  preview.classList.add("hidden");
  dropInner.classList.remove("hidden");
  analyzeBtn.disabled = true;

  resultCard.classList.add("hidden");
  uploadCard.classList.remove("hidden");
  hideError();
});

// ── HELPERS ───────────────────────────────────────────────────────
function setLoading(state) {
  analyzeBtn.disabled = state;
  btnText.textContent = state ? "Analysing…" : "Analyse Scan";
  btnLoader.classList.toggle("hidden", !state);
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}
