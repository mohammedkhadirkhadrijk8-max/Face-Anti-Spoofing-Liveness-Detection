# Face Anti-Spoofing & Liveness Detection

<p align="center">
  <strong>A PyTorch research prototype for classifying face images and video frames as bona fide or presentation attacks.</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-64748B">
</p>

This repository contains a multi-stream face anti-spoofing prototype, training and evaluation code, image/video/webcam inference, and generated project materials. It combines RGB appearance features with Local Binary Pattern (LBP) texture and YCrCb color features.

> **Important evaluation note:** The included sample data and reported example scores come from a small, generated synthetic dataset. They are useful for demonstrating that the pipeline runs, but they are **not** evidence of accuracy on real people, real presentation attacks, or deployment conditions. This project is a research/learning prototype, not a certified biometric security product.

## What it does

- Classifies an input frame as `LIVE` (bona fide) or `SPOOF` (presentation attack).
- Supports still-image, video-file, and webcam inference.
- Optionally detects faces with OpenCV's Haar cascade before classifying each detected crop.
- Trains a PyTorch model from a folder-based image dataset, with a synthetic fallback for smoke testing.
- Computes EER, APCER, BPCER, HTER, and a confusion matrix from bona fide class scores.
- Includes a PDF technical report, a PowerPoint presentation, and a training-results plot.

This is **presentation-attack classification**, not facial identity recognition. The video mode processes frames independently; it does not perform temporal liveness challenges, pulse analysis, depth sensing, or camera-stream injection detection.

## Model architecture

The network extracts three complementary feature representations and concatenates them before binary classification:

```text
RGB 128x128 ── MobileNetV3-Small ────────── 576 features ─┐
LBP 128x128 ── lightweight 4-stage CNN ─────  64 features ─┼─ 704 features
YCrCb 128x128 ─ lightweight 4-stage CNN ───── 64 features ─┘
                                  │
                 FC(256) → BN → ReLU → Dropout(0.4)
                 FC(128) → BN → ReLU → Dropout(0.2)
                 FC(2) → [SPOOF, LIVE]
```

The RGB branch uses ImageNet-pretrained MobileNetV3-Small by default during training. LBP captures local intensity texture; YCrCb provides a luminance/chrominance representation. The side branches and fusion head are trained with the backbone.

## Repository contents

| Path | Purpose |
| --- | --- |
| `dataset.py` | Image discovery, preprocessing, LBP/YCrCb extraction, augmentation, and synthetic fallback |
| `model.py` | Multi-stream `FaceAntiSpoofNet` architecture |
| `train.py` | Training loop, checkpoint saving, and training/evaluation plot generation |
| `inference.py` | Image, video, and webcam prediction CLI |
| `metrics.py` | EER, APCER, BPCER, HTER, and confusion-matrix calculations |
| `generate_pdf_report.py` | Source for the 10-page PDF report |
| `generate_ppt.py` | Source for the PowerPoint presentation |
| `run_live_demo.command` | macOS terminal launcher for webcam inference |
| `synthetic_data/` | 400 generated `real` and 400 generated `spoof` example images |
| `best_antispoof_model.pth` | Example trained model checkpoint |
| `results.png` | Example training curves and ROC plot |
| `Face_Anti_Spoofing_Project_Report.pdf` | Generated technical report |
| `Face_Anti_Spoofing_Presentation.pptx` | Generated slide deck |

## Quick start

### 1. Set up Python

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run an image prediction

The included checkpoint is used by default:

```bash
python inference.py --image path/to/image.jpg --no-display
```

For an annotated preview window and saved output, omit `--no-display`:

```bash
python inference.py --image path/to/image.jpg
```

### 3. Run video or webcam inference

```bash
python inference.py --video path/to/video.mp4
python inference.py                       # default webcam (camera 0)
```

Press `q` in the video window to exit. On macOS, grant camera access to the terminal or IDE running Python. The included `run_live_demo.command` is a macOS convenience launcher.

### 4. Train

```bash
python train.py
```

Training looks for images under the `dataset_root` default configured in `dataset.py`. If that location is missing or empty, it generates/uses `synthetic_data/` so the pipeline can be smoke-tested. To use your own dataset, update the `dataset_root` default in `build_dataloaders` in `dataset.py` to its location.

Expected layout (folder names are case-sensitive and must use one of the supported class names in `dataset.py`):

```text
your_dataset/
├── real/       # or live/, bonafide/, genuine/, ...
│   ├── sample_001.jpg
│   └── ...
└── spoof/      # or fake/, attack/, print/, replay/, mask/, ...
    ├── sample_001.jpg
    └── ...
```

Training writes `best_antispoof_model.pth` and `results.png` in the project directory. The first training run may download MobileNetV3 pretrained weights through `torchvision`.

### 5. Regenerate project materials

```bash
python generate_pdf_report.py
python generate_ppt.py
```

The PowerPoint generator uses `python-pptx`, included in `requirements.txt`.

## Evaluation metrics

`metrics.py` expects labels `0 = spoof/attack` and `1 = bona fide/real`, plus model scores for class 1. It reports:

- **EER** — the point on the ROC curve where false-accept and false-reject rates are closest.
- **APCER** — attack samples classified as bona fide divided by attack samples.
- **BPCER** — bona fide samples classified as attacks divided by bona fide samples.
- **HTER** — the mean of APCER and BPCER.

The training pipeline calculates these on its held-out split at the EER-derived threshold. A metric implementation does not by itself establish ISO/IEC 30107-3 conformance or certification. Meaningful performance claims require a suitable real-world dataset, protocol, subject-disjoint splits, attack coverage, and independent testing.

## Limitations and responsible use

- The bundled generated images are simplistic synthetic examples, not a representative face or attack dataset. A high score on this split should not be interpreted as generalization.
- The default split shuffles individual images; it does not enforce subject-disjoint, session-disjoint, or device-disjoint evaluation.
- The model has not been validated against public anti-spoofing benchmarks or unknown attack types, lighting, cameras, or demographic groups.
- Predictions are not calibrated security guarantees. Do not use this prototype as the sole control for account access, identity verification, payments, or other high-impact decisions.
- Face detection is optional in the inference flow. If no face is detected (or detection is unavailable), the full image/frame may be classified instead.
- The project does not identify a person and does not provide active liveness challenges or deepfake/injection defenses.

For security-sensitive use, replace the sample data, define a rigorous evaluation protocol, test under the deployment domain, and add independent safeguards and human review.

## Project materials

- [Technical report (PDF)](Face_Anti_Spoofing_Project_Report.pdf)
- [Presentation (PPTX)](Face_Anti_Spoofing_Presentation.pptx)
- [Example training plot](results.png)

## License

No license is currently included. All rights and reuse permissions remain with the respective rights holders unless a license is added.
