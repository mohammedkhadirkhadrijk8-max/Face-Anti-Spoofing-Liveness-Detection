"""
inference.py  –  Real-Time Liveness Detection Inference
=======================================================
Modes:
    1. Webcam   :  python inference.py                     (default)
    2. Image    :  python inference.py --image path/to/img.jpg
    3. Video    :  python inference.py --video path/to/vid.mp4

Loads  best_antispoof_model.pth  and displays "LIVE" / "SPOOF" with
confidence scores on every frame.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

# Local modules
from model   import FaceAntiSpoofNet
from dataset import compute_lbp, rgb_to_ycrcb

# ──────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "best_antispoof_model.pth")
IMG_SIZE   = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
#  Preprocessing (mirrors dataset.py but for a single frame)
# ──────────────────────────────────────────────────────────────────────────
_rgb_normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


def preprocess_frame(frame_bgr: np.ndarray):
    """
    Convert an OpenCV BGR frame into the three model inputs.

    Returns
    -------
    rgb_t   : (1, 3, 128, 128) tensor
    ycrcb_t : (1, 3, 128, 128) tensor
    lbp_t   : (1, 1, 128, 128) tensor
    """
    # BGR → RGB
    frame_rgb = frame_bgr[:, :, ::-1].copy()
    # Resize
    img_pil = Image.fromarray(frame_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np  = np.array(img_pil, dtype=np.uint8)

    # RGB tensor
    rgb_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    rgb_t = _rgb_normalize(rgb_t).unsqueeze(0)

    # YCrCb tensor
    ycrcb_np = rgb_to_ycrcb(img_np)
    ycrcb_t  = torch.from_numpy(ycrcb_np).permute(2, 0, 1).unsqueeze(0)

    # LBP tensor
    gray   = np.mean(img_np.astype(np.float32), axis=2)
    lbp_np = compute_lbp(gray.astype(np.uint8))
    lbp_t  = torch.from_numpy(lbp_np).unsqueeze(0).unsqueeze(0)

    return rgb_t.to(DEVICE), ycrcb_t.to(DEVICE), lbp_t.to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────
#  Load model
# ──────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH):
    model = FaceAntiSpoofNet(backbone="mobilenet_v3_small", pretrained=False)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        print("        Run train.py first to generate best_antispoof_model.pth")
        sys.exit(1)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[inference] Model loaded from {model_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────
#  Predict on a single frame
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, frame_bgr: np.ndarray):
    """
    Returns
    -------
    label      : str  ("LIVE" or "SPOOF")
    confidence : float  (probability of the predicted class)
    prob_real  : float  (probability for bonafide class)
    """
    rgb_t, ycrcb_t, lbp_t = preprocess_frame(frame_bgr)
    logits = model(rgb_t, ycrcb_t, lbp_t)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label      = "LIVE" if pred_class == 1 else "SPOOF"
    prob_real  = float(probs[1])
    return label, confidence, prob_real


# ──────────────────────────────────────────────────────────────────────────
#  Webcam / video loop
# ──────────────────────────────────────────────────────────────────────────

def run_video(model, source=0):
    """Real-time liveness detection on webcam (source=0) or a video file."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) is required for video inference.")
        print("        Install it:  pip install opencv-python")
        sys.exit(1)

    # Try default backend first, fallback to AVFOUNDATION on macOS
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        print("\n💡 macOS Camera Troubleshooting:")
        print("   macOS blocks camera access for CLI apps until permitted.")
        print("   1. Open 'System Settings' → 'Privacy & Security' → 'Camera'")
        print("   2. Turn ON the toggle for Terminal / Visual Studio Code / Python")
        print("   3. If you just changed permissions, restart your terminal or IDE.")
        sys.exit(1)

    print("[inference] Press 'q' to quit.\n")

    # Load Haar cascade for face detection if available
    face_cascade = None
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        try:
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
        except Exception:
            face_cascade = None

    import time
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        # Detect faces if cascade is available
        faces = ()
        if face_cascade is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                   minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            # Analyze frame
            label, conf, _ = predict(model, frame)
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            text = f"Status: {label} ({conf*100:.1f}%)"
            # Background rectangle for readable text
            cv2.rectangle(display, (10, 15), (420, 65), (0, 0, 0), -1)
            cv2.putText(display, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            for (x, y, w, h) in faces:
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_crop = frame[y1:y2, x1:x2]

                label, conf, _ = predict(model, face_crop)
                color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                text = f"{label} {conf*100:.1f}%"
                cv2.rectangle(display, (x1, max(0, y1 - 35)), (x1 + 200, y1), color, -1)
                cv2.putText(display, text, (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS indicator
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Anti-Spoofing – Real-Time Liveness", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[inference] Done.")


# ──────────────────────────────────────────────────────────────────────────
#  Single image inference
# ──────────────────────────────────────────────────────────────────────────

def run_image(model, image_path: str, no_display: bool = False):
    """Run liveness prediction on a single image file with face detection."""
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    if not has_cv2 or no_display:
        img_pil = Image.open(image_path).convert("RGB")
        img_np  = np.array(img_pil)[:, :, ::-1]  # RGB→BGR
        label, conf, prob = predict(model, img_np)
        print(f"\n  Image : {image_path}")
        print(f"  Result: {label}  (confidence: {conf*100:.1f}%, P(real): {prob*100:.1f}%)\n")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    display = img.copy()

    # Detect faces if Haar cascade is available
    faces = ()
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        try:
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                   minNeighbors=4, minSize=(60, 60))
        except Exception:
            faces = ()

    if len(faces) == 0:
        label, conf, prob = predict(model, img)
        color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
        text  = f"{label}  ({conf*100:.1f}%)"
        cv2.putText(display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        print(f"\n  [No isolated face detected; analyzed full frame]")
        print(f"  Image : {image_path}")
        print(f"  Result: {label}  (confidence: {conf*100:.1f}%, P(real): {prob*100:.1f}%)\n")
    else:
        print(f"\n  [Detected {len(faces)} face(s) in image]")
        for i, (x, y, w, h) in enumerate(faces):
            margin = int(0.2 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)
            face_crop = img[y1:y2, x1:x2]

            label, conf, prob = predict(model, face_crop)
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            text  = f"{label} {conf*100:.1f}%"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            cv2.putText(display, text, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            print(f"  Face #{i+1}: {label}  (confidence: {conf*100:.1f}%, P(real): {prob*100:.1f}%)")

    # Save output visualization
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_output.jpg")
    cv2.imwrite(out_path, display)
    print(f"\n  Annotated image saved → {out_path}")

    # Display image with automatic 4-second close so terminal doesn't hang
    try:
        cv2.imshow("Face Anti-Spoofing – Liveness Detection", display)
        print("  Displaying window (auto-closes in 4s or press any key) …\n")
        cv2.waitKey(4000)
        cv2.destroyAllWindows()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face Anti-Spoofing – Liveness Detection Inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image file.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file (or 0 for webcam).")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to trained model weights (.pth).")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable GUI window display (useful for headless/testing).")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        run_image(model, args.image, no_display=args.no_display)
    elif args.video:
        source = int(args.video) if args.video.isdigit() else args.video
        run_video(model, source)
    else:
        # Default: webcam
        print("[inference] Starting webcam liveness detection …")
        run_video(model, source=0)


if __name__ == "__main__":
    main()
