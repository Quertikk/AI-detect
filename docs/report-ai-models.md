# Project Report — Synthetic Face Detection via Transfer Learning

**Subject:** AI Models Pre-training and Training in Digital Reality Environments and 3D/XR Simulators Building
**Author:** Valentyn Hotsulenko
**Student ID:** 494098
**Programme:** Technologie Komputerowe
**Repository:** https://github.com/Quertikk/AI-detect

---

## 1. Problem statement

Generative models now produce human faces that people cannot reliably distinguish from photographs. This has direct consequences wherever a face is used as evidence of identity: disinformation, fraud, and — relevant to digital reality environments — the authenticity of the person behind a photorealistic avatar.

The practical question this project addresses is narrow and answerable: **given an image of a face, can a model trained on a modest budget decide whether that face was photographed or generated, and explain its decision well enough for a human to act on it?**

## 2. Objectives and scope

| In scope | Out of scope |
|---|---|
| Binary real/synthetic classification of face images | General "was this image AI-generated" detection for arbitrary content |
| Transfer learning from an ImageNet-pre-trained backbone | Training a generative model, or training a backbone from scratch |
| Explainability of each verdict (Grad-CAM, frequency analysis) | Rendering, simulation, or building of 3D/XR environments |
| Frame-level extension to video | A dedicated temporal/video architecture |

**A deliberate honesty note on subject fit.** This project demonstrates the *pre-training → fine-tuning* pipeline that the subject is concerned with, and the detection problem is directly motivated by identity in digital reality environments (photorealistic avatars, synthetic profile imagery). It does **not** build a 3D/XR simulator, and nothing in this report should be read as claiming otherwise.

## 3. Method

### 3.1 Why transfer learning rather than training from scratch

Training a competitive vision backbone from scratch requires millions of labelled images and GPU-weeks. The available hardware was a single laptop RTX 3050 with 4 GB of VRAM. Transfer learning resolves this: a backbone pre-trained on ImageNet has already learned general visual features (edges, textures, facial structure), and only the task-specific decision layer needs to be learned.

This is the core trade-off the subject is about, so it is worth stating precisely:

- **Pre-training** (done for us, on ImageNet-1k): learns general-purpose features from a large, diverse corpus.
- **Fine-tuning** (done here, on 12,000 face images): adapts the final layers to separate real from synthetic faces.

### 3.2 Architecture

**Backbone:** EfficientNet-B0 via `timm`, ImageNet-pre-trained. Chosen for its accuracy-per-parameter ratio — roughly 5.3M parameters, which fits comfortably in 4 GB of VRAM at batch size 32.

**Freezing strategy:** all parameters frozen except:
- `conv_head` (final convolutional block)
- `bn2` (its batch-norm)
- `classifier` (the new single-logit head)

The head outputs one logit; `sigmoid(logit)` is interpreted as P(real), trained with `BCEWithLogitsLoss`. Freezing the early layers both prevents overfitting on a small dataset and cuts backward-pass cost, which is what brings training time down to minutes.

### 3.3 Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) — real faces from Flickr-FHQ, synthetic faces from StyleGAN.

The full set was deliberately subsampled to keep the training budget small:

| Split | Real | Synthetic | Total |
|---|---|---|---|
| Train | 6,000 | 6,000 | 12,000 |
| Validation | 1,200 | 1,200 | 2,400 |
| Test | 1,200 | 1,200 | 2,400 |

Classes are balanced, so accuracy is a meaningful headline metric. The test split is held out and touched only once, after model selection.

### 3.4 Training protocol

| Parameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Normalisation | ImageNet mean/std |
| Augmentation (train only) | random horizontal flip, colour jitter (brightness 0.15, contrast 0.15, saturation 0.1) |
| Optimiser | AdamW, lr = 3 × 10⁻⁴ |
| Loss | `BCEWithLogitsLoss` |
| Batch size | 32 |
| Epochs | 4 |
| Model selection | best validation F1 |

Colour jitter is included on purpose: it discourages the model from latching onto global colour statistics, which differ between the two source corpora and would be a shortcut rather than a real artefact cue.

## 4. Results

### 4.1 Training progression

| Epoch | Train loss | Val accuracy | Val F1 | Time |
|---|---|---|---|---|
| 1 | 0.8809 | 0.8325 | 0.8388 | 116 s |
| 2 | 0.4191 | 0.8508 | 0.8460 | 59 s |
| 3 | 0.2903 | 0.8696 | 0.8688 | 56 s |
| 4 | 0.2092 | 0.8758 | 0.8749 | 57 s |

Validation metrics were still improving at epoch 4, so the model is under-trained rather than over-fitted — more epochs would likely help. Training took **under 5 minutes in total**.

### 4.2 Held-out test performance

Best checkpoint, evaluated once on 2,400 unseen images (`models/metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | 88.13% |
| Precision | 88.54% |
| Recall | 87.58% |
| F1 | 0.8806 |
| ROC-AUC | 0.9479 |

ROC-AUC of 0.948 against accuracy of 0.881 indicates the model ranks samples considerably better than the fixed 0.5 threshold captures — a deployment could tune that threshold to favour precision or recall depending on the cost of each error type.

### 4.3 Explainability

A bare probability is not actionable, so each verdict is accompanied by two artefacts:

**Grad-CAM** over `conv_head`, producing a heatmap of the regions that most influenced the output. Stated precisely: this shows *where the model looked*, which is not the same as a verified artefact location. It is an aid to human judgement, not proof.

**FFT log-magnitude spectrum.** GAN upsampling can leave periodic artefacts in the frequency domain that are invisible in pixel space. This is computed analytically — it is **not** a trained component and does **not** contribute to the verdict. It is displayed as supporting context only.

### 4.4 Extension to video

No temporal model was trained. Video is handled by sampling one frame per second (capped at 30 frames), detecting the largest face per frame with an OpenCV Haar cascade, scoring each crop with the same classifier, and averaging. A face bounding-box jitter statistic is also reported, clearly labelled experimental and excluded from the verdict — camera shake in genuine footage produces high jitter too.

## 5. Limitations

1. **Face-specific.** Trained on faces; it is not a general AI-image detector.
2. **One generator family.** The synthetic half is StyleGAN. Performance against diffusion models or unseen generators is untested and would likely be worse — this is the standard generalisation gap in this literature.
3. **Frame-level video.** Temporal inconsistency, the strongest deepfake-video cue, is not modelled.
4. **Haar cascade face detection** misses off-angle and partially occluded faces.
5. **Under-trained by design.** Metrics were still climbing when training stopped.
6. **No audio.**

## 6. Conclusions and further work

A competitive face-authenticity classifier was produced in under five minutes of GPU time on consumer hardware, reaching 88.1% accuracy and 0.948 ROC-AUC on a held-out set. The result supports the central claim of the transfer-learning approach: for a narrow task with a modest dataset, adapting a pre-trained backbone is far more efficient than training one.

Worthwhile next steps, in order of expected value:

1. Add diffusion-generated faces to the training set to close the main generalisation gap.
2. Train a temporal model (3D CNN or CNN+LSTM) for video rather than averaging frames.
3. Unfreeze more of the backbone with a discriminative learning rate.
4. Calibrate the decision threshold against an explicit cost model.

## 7. Reproducing this work

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python training/dataset_prep.py --per-class 6000    # needs a Kaggle API token
python training/train.py --data data/faces --epochs 4 --batch-size 32
cd backend && uvicorn app:app
```

Relevant source: `training/train.py` (training), `backend/model.py` (inference + Grad-CAM), `backend/fft_analysis.py` (spectrum), `backend/video.py` (video pipeline).

## 8. References

- Tan, M. & Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- Selvaraju, R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Karras, T. et al. *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR 2019.
- Wang, S.-Y. et al. *CNN-generated images are surprisingly easy to spot… for now.* CVPR 2020.
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), Kaggle.
