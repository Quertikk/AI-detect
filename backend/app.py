"""
FastAPI backend for the synthetic media detector.

Run with:
    uvicorn backend.app:app --reload

Serves the static frontend at "/" and the API under "/api/...".
Analysis results are kept in an in-memory store keyed by an id so the PDF
report endpoint can look them back up - fine for a single-process course
demo, not meant to survive a restart or scale beyond it.
"""
import base64
import io
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from charts import render_timeline_png
from fft_analysis import compute_fft_spectrum
from model import DeepfakeClassifier
from report import generate_report
from video import VideoAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
MAX_STORED_ANALYSES = 50

app = FastAPI(title="Synthetic Media Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: DeepfakeClassifier | None = None
video_analyzer: VideoAnalyzer | None = None
analyses: "OrderedDict[str, dict]" = OrderedDict()


@app.on_event("startup")
def load_model() -> None:
    global classifier, video_analyzer
    classifier = DeepfakeClassifier()
    video_analyzer = VideoAnalyzer(classifier)


def _store(analysis: dict) -> str:
    analysis_id = str(uuid.uuid4())
    analyses[analysis_id] = analysis
    while len(analyses) > MAX_STORED_ANALYSES:
        analyses.popitem(last=False)
    return analysis_id


def _png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(503, "Model not loaded yet")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")

    prediction = classifier.predict(image)
    heatmap_png = _image_to_png_bytes(classifier.gradcam_heatmap(image))
    fft_png = _image_to_png_bytes(compute_fft_spectrum(image))

    analysis = {
        "kind": "image",
        "filename": file.filename,
        "verdict": prediction["label"],
        "confidence": prediction["confidence"],
        "prob_fake": prediction["prob_fake"],
        "heatmap_png": heatmap_png,
        "fft_png": fft_png,
    }
    analysis_id = _store(analysis)

    return {
        "id": analysis_id,
        "verdict": prediction["label"],
        "confidence": prediction["confidence"],
        "prob_fake": prediction["prob_fake"],
        "heatmap": _png_to_data_uri(heatmap_png),
        "fft": _png_to_data_uri(fft_png),
    }


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if video_analyzer is None:
        raise HTTPException(503, "Model not loaded yet")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    raw = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        result = video_analyzer.analyze(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if "error" in result:
        raise HTTPException(422, result["error"])

    timeline_png = render_timeline_png(result["frame_results"])

    analysis = {
        "kind": "video",
        "filename": file.filename,
        "verdict": result["verdict"],
        "avg_prob_fake": result["avg_prob_fake"],
        "suspicious_frame_ratio": result["suspicious_frame_ratio"],
        "temporal_jitter_score": result["temporal_jitter_score"],
        "timeline_png": timeline_png,
    }
    analysis_id = _store(analysis)

    return {
        "id": analysis_id,
        "verdict": result["verdict"],
        "avg_prob_fake": result["avg_prob_fake"],
        "suspicious_frame_ratio": result["suspicious_frame_ratio"],
        "temporal_jitter_score": result["temporal_jitter_score"],
        "frame_results": result["frame_results"],
        "timeline": _png_to_data_uri(timeline_png),
    }


@app.get("/api/report/{analysis_id}")
def get_report(analysis_id: str):
    analysis = analyses.get(analysis_id)
    if analysis is None:
        raise HTTPException(404, "Unknown or expired analysis id")

    pdf_bytes = generate_report(analysis)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{analysis_id[:8]}.pdf"'},
    )


frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
