const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const verdictBadge = document.getElementById("verdict-badge");
const confidenceText = document.getElementById("confidence-text");
const reportBtn = document.getElementById("report-btn");
const imagePanels = document.getElementById("image-panels");
const videoPanels = document.getElementById("video-panels");
const heatmapImg = document.getElementById("heatmap-img");
const fftImg = document.getElementById("fft-img");
const timelineImg = document.getElementById("timeline-img");
const videoDetails = document.getElementById("video-details");

let currentAnalysisId = null;

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function setStatus(text) {
  statusEl.hidden = !text;
  statusEl.textContent = text || "";
}

function handleFile(file) {
  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");
  if (!isVideo && !isImage) {
    setStatus("Unsupported file type.");
    return;
  }

  resultEl.hidden = true;
  reportBtn.hidden = true;
  setStatus(`Analyzing ${file.name}...`);

  const formData = new FormData();
  formData.append("file", file);
  const endpoint = isVideo ? "/api/analyze/video" : "/api/analyze/image";

  fetch(endpoint, { method: "POST", body: formData })
    .then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      return res.json();
    })
    .then((data) => (isVideo ? renderVideoResult(data) : renderImageResult(data)))
    .catch((err) => setStatus(`Error: ${err.message}`))
    .finally(() => setStatus(""));
}

function showVerdict(verdict, confidence) {
  verdictBadge.textContent = verdict;
  verdictBadge.className = `badge ${verdict}`;
  confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  resultEl.hidden = false;
}

function renderImageResult(data) {
  currentAnalysisId = data.id;
  showVerdict(data.verdict, data.confidence);
  imagePanels.hidden = false;
  videoPanels.hidden = true;
  heatmapImg.src = data.heatmap;
  fftImg.src = data.fft;
  reportBtn.hidden = false;
}

function renderVideoResult(data) {
  currentAnalysisId = data.id;
  showVerdict(data.verdict, data.avg_prob_fake >= 0.5 ? data.avg_prob_fake : 1 - data.avg_prob_fake);
  imagePanels.hidden = true;
  videoPanels.hidden = false;
  timelineImg.src = data.timeline;

  videoDetails.innerHTML = "";
  const items = [
    ["Average P(fake)", `${(data.avg_prob_fake * 100).toFixed(1)}%`],
    ["Suspicious frames", `${(data.suspicious_frame_ratio * 100).toFixed(1)}%`],
    ["Frames analyzed", `${data.frame_results.length}`],
  ];
  if (data.temporal_jitter_score !== null && data.temporal_jitter_score !== undefined) {
    items.push(["Temporal jitter score (experimental)", data.temporal_jitter_score.toFixed(4)]);
  }
  for (const [label, value] of items) {
    const li = document.createElement("li");
    li.textContent = `${label}: ${value}`;
    videoDetails.appendChild(li);
  }
  reportBtn.hidden = false;
}

reportBtn.addEventListener("click", () => {
  if (!currentAnalysisId) return;
  window.location.href = `/api/report/${currentAnalysisId}`;
});
