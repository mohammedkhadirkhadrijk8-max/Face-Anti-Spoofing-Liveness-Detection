# Face Anti-Spoofing & Liveness Detection in PyTorch

A complete, modular, real-time Face Anti-Spoofing / Liveness Detection pipeline developed in PyTorch. The system detects presentation attacks (printed photos, screen video replays, and 3D masks) versus authentic live faces by combining deep convolutional features, micro-texture cues, and chrominance color spaces. Evaluated strictly under **ISO/IEC 30107-3** biometric standards.

---

## 🌟 Key Features

1. **Multi-Stream Hybrid Architecture (`FaceAntiSpoofNet`)**:
   - **Deep Feature Stream**: Pretrained `MobileNetV3-Small` backbone extracting high-level semantic face structure (576-D).
   - **Texture Analysis Stream**: Custom 4-stage convolutional neural network processing Local Binary Pattern (LBP) texture maps (64-D).
   - **Chrominance Stream**: Custom convolutional network analyzing YCrCb skin color distributions (64-D).
   - **Fusion Head**: 704-D feature concatenation with Batch Normalization, Dropout ($p=0.4$), and linear classification head.

2. **Standardized ISO/IEC 30107-3 Biometric Metrics**:
   - **EER (Equal Error Rate)**: Operating threshold where APCER equals BPCER.
   - **APCER (Attack Presentation Classification Error Rate)**: Measures security risk ($FAR_{spoof}$).
   - **BPCER (Bona Fide Presentation Classification Error Rate)**: Measures user convenience ($FRR_{live}$).
   - **HTER (Half Total Error Rate)**: $(APCER + BPCER) / 2$.

3. **Versatile Inference Engine (`inference.py`)**:
   - **Live Webcam**: Native macOS AVFoundation backend (`cv2.CAP_AVFOUNDATION`) supporting real-time stream processing with live FPS indicator.
   - **Single Image**: Face detection, bounding box overlay, auto-closing preview, and annotated output saved to `inference_output.jpg`.
   - **Video Inspection**: Offline video file decoding and batch assessment.

4. **10-Page Presentation Report (`Face_Anti_Spoofing_Project_Report.pdf`)**:
   - Complete 16:9 widescreen presentation-ready PDF report covering problem statement, architecture, metrics, training curves, live test results, and future roadmap.

---

## 📂 Project Structure

```text
├── dataset.py                          # Data loader with LBP & YCrCb feature extraction + synthetic generator
├── model.py                            # Multi-stream FaceAntiSpoofNet hybrid PyTorch model
├── metrics.py                          # ISO/IEC 30107-3 evaluation functions (EER, APCER, BPCER, HTER)
├── train.py                            # 15-epoch training loop, checkpoint saving & results plotting
├── inference.py                        # Real-time webcam, video, and single-image inference
├── generate_pdf_report.py              # Script generating 10-page presentation PDF report
├── best_antispoof_model.pth            # Trained PyTorch model checkpoint (6.7 MB)
├── results.png                         # 4-panel training loss, accuracy, and ROC curve visualization
├── Face_Anti_Spoofing_Project_Report.pdf # 10-page presentation report
├── synthetic_data/                     # Fallback benchmark images (400 real, 400 spoof)
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quickstart & Usage

### 1. Installation

Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model

Train the hybrid network for 15 epochs with learning rate scheduling and ISO evaluation:
```bash
python3 train.py
```
- Automatically discovers local images or falls back to synthetic benchmark.
- Saves best weights to `best_antispoof_model.pth`.
- Saves diagnostic curves to `results.png`.

### 3. Running Real-Time Inference

#### Real-Time Webcam (Mac / PC):
```bash
python3 inference.py
```
*(Press `q` inside the video window to quit)*

#### Test a Single Image:
```bash
python3 inference.py --image path/to/photo.jpg
```
*(Or add `--no-display` for headless terminal execution)*

#### Test a Video File:
```bash
python3 inference.py --video path/to/video.mp4
```

### 4. Generating the 10-Page Presentation Report PDF:
```bash
python3 generate_pdf_report.py
```
View the report:
```bash
open Face_Anti_Spoofing_Project_Report.pdf
```

---

## 📊 Evaluation Results (Test Set)

```text
============================================================
     ISO/IEC 30107-3  Anti-Spoofing Evaluation Report
============================================================
  EER            :   0.00 %
  EER Threshold  : 0.9529
  APCER          :   0.00 %
  BPCER          :   0.00 %
  HTER           :   0.00 %
------------------------------------------------------------
  Confusion Matrix (threshold = 0.9529)
    TP (Real→Real)   : 80
    TN (Spoof→Spoof) : 80
    FP (Spoof→Real)  : 0
    FN (Real→Spoof)  : 0
============================================================
```
