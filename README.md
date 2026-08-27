# Synthetic Media Detector

A web tool that checks whether a face photo or short video is real or AI-generated / face-swapped, and explains *why* — instead of returning a bare yes/no.

## Problem, need, solution

- **Problem**: AI-generated and face-swapped media (diffusion images, deepfake video) is now hard to tell apart from real footage by eye, and is actively used for disinformation and fraud.
- **Need**: journalists, OSINT investigators and everyday users need a fast, explainable check — not a black box, something they can point to when writing up a finding.
- **Solution**: upload an image or video, get a verdict plus supporting evidence: a Grad-CAM heatmap of what the model focused on, an FFT spectrum panel, and (for video) a per-frame timeline — bundled into a downloadable PDF report.

## Architecture

```
deepfake-detector/
  backend/        FastAPI app: inference, Grad-CAM, FFT, video pipeline, PDF report
  training/       Dataset prep + fine-tuning script for the classifier
  frontend/       Static single-page UI (vanilla HTML/CSS/JS, no build step)
  models/         Trained checkpoint + metrics (produced by training/train.py)
```

**Classifier**: EfficientNet-B0 (ImageNet-pretrained via `timm`), fine-tuned as a binary real/fake face classifier. Only the classifier head and the last backbone block are unfrozen, so fine-tuning is fast even on a small GPU.

**Video handling**: no separate video model is trained. Frames are sampled at a fixed interval, a face is detected per frame (OpenCV Haar cascade), and each face crop is scored by the same image classifier. Scores are aggregated into a video-level verdict, plus an experimental face-bounding-box jitter heuristic reported for context (not used in the verdict).

**Explainability, not just a score**:
- Grad-CAM overlay showing which image regions most influenced the model's output.
- FFT log-magnitude spectrum — a fixed signal-processing visualization (not a trained model) shown to support the verdict, since GAN/diffusion upsampling can leave frequency-domain artifacts.
- PDF report bundling verdict, confidence, and both visualizations.

## Getting started

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the CPU/CUDA build matching your machine
pip install -r requirements.txt
```

### 2. Get training data

Requires a free Kaggle account and API token (`kaggle.json` from https://www.kaggle.com/settings, placed at `~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars):

```bash
python training/dataset_prep.py --per-class 5000
```

Downloads the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset and writes a subsampled train/valid/test split to `data/faces/`.

### 3. Train the classifier

```bash
python training/train.py --data data/faces --epochs 3 --batch-size 32
```

Saves the best checkpoint to `models/classifier.pt` and test-set metrics to `models/metrics.json`.

### 4. Run the app

```bash
cd backend
uvicorn app:app --reload
```

Open http://localhost:8000 — upload an image or video and get a verdict with a Grad-CAM heatmap, FFT panel, and (for video) a per-frame timeline. Download a PDF report from the result panel.

## Measured results

Filled in after running `training/train.py` — see `models/metrics.json` for the exact numbers (accuracy / precision / recall / F1 / ROC-AUC on the held-out test split of the subsampled dataset).

## Limitations (by design, not oversight)

- **Face-only**: the classifier is trained on a face-swap/GAN face dataset, not a general "is this AI-generated" detector for arbitrary images.
- **Video is frame-level, not temporal**: each frame is scored independently by the image classifier; there is no dedicated video/temporal model. This is a deliberate scope decision to keep training time small.
- **Temporal jitter score is experimental**: it measures face bounding-box position/size fluctuation across sampled frames and is reported for context only — it is not used to compute the verdict, and a real video with camera shake can also score high on it.
- **No audio analysis.**
- **Generalization is bounded by training data**: the classifier will be less reliable on generators/manipulation techniques not represented in its training set.
- **In-memory analysis store**: the backend keeps recent results in memory (capped, most-recent-evicted) so the PDF report endpoint can look them up; this is fine for a live demo, not a production deployment.
