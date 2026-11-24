#!/usr/bin/env python3
"""
generate_ppt.py  –  Generate a professional 4-slide class presentation
======================================================================
Creates  Face_Anti_Spoofing_Presentation.pptx  in the project directory.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SAVE_DIR, "Face_Anti_Spoofing_Presentation.pptx")

# ── Colour palette ──
BG_DARK   = RGBColor(0x0F, 0x17, 0x2A)   # deep navy
BG_CARD   = RGBColor(0x1A, 0x25, 0x3C)   # card background
ACCENT    = RGBColor(0x00, 0xD2, 0xFF)    # cyan accent
ACCENT2   = RGBColor(0x7C, 0x3A, 0xED)   # purple accent
GREEN     = RGBColor(0x10, 0xB9, 0x81)    # success green
RED       = RGBColor(0xEF, 0x44, 0x44)    # alert red
ORANGE    = RGBColor(0xF5, 0x9E, 0x0B)    # warning orange
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT     = RGBColor(0xCB, 0xD5, 0xE1)    # light grey text
MUTED     = RGBColor(0x94, 0xA3, 0xB8)    # muted text


def set_slide_bg(slide, color):
    """Set slide background to a solid colour."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, corner_radius=None):
    """Add a rounded rectangle shape."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()  # no border
    if corner_radius is not None:
        shape.adjustments[0] = corner_radius
    return shape


def add_text_box(slide, left, top, width, height):
    """Add a text box and return the text frame."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    return tf


def set_para(tf, text, size=18, color=WHITE, bold=False, alignment=PP_ALIGN.LEFT,
             space_before=0, space_after=0, font_name="Calibri"):
    """Add a paragraph to a text frame."""
    p = tf.add_paragraph() if len(tf.paragraphs) > 0 and tf.paragraphs[0].text != "" else tf.paragraphs[0]
    if tf.paragraphs[0].text != "" or len(tf.paragraphs) > 1:
        p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    return p


def add_bullet(tf, text, size=14, color=LIGHT, indent_level=0, bold=False,
               space_before=4, font_name="Calibri"):
    """Add a bullet point."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.level = indent_level
    p.space_before = Pt(space_before)
    p.space_after = Pt(2)
    return p


# ══════════════════════════════════════════════════════════════════════
#  CREATE PRESENTATION
# ══════════════════════════════════════════════════════════════════════

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height

# ─────────────────────────────────────────────────────────────────────
#  SLIDE 1 — Title Slide
# ─────────────────────────────────────────────────────────────────────
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide1, BG_DARK)

# Decorative accent bar at top
add_shape(slide1, Inches(0), Inches(0), SLIDE_W, Inches(0.06), ACCENT)

# Left accent strip
add_shape(slide1, Inches(0), Inches(0), Inches(0.08), SLIDE_H, ACCENT2)

# Main title area
tf = add_text_box(slide1, Inches(1.2), Inches(1.5), Inches(10.5), Inches(1.2))
set_para(tf, "🔐  FACE ANTI-SPOOFING", size=44, color=ACCENT, bold=True,
         alignment=PP_ALIGN.LEFT)

tf2 = add_text_box(slide1, Inches(1.2), Inches(2.8), Inches(10.5), Inches(1.0))
set_para(tf2, "& Liveness Detection System", size=36, color=WHITE, bold=True)

# Subtitle
tf3 = add_text_box(slide1, Inches(1.2), Inches(4.0), Inches(10), Inches(0.7))
set_para(tf3, "A Deep Learning Approach Using Multi-Stream Hybrid CNN Architecture  |  PyTorch Implementation",
         size=18, color=LIGHT)

# Divider line
add_shape(slide1, Inches(1.2), Inches(5.0), Inches(4), Inches(0.04), ACCENT)

# Details
tf4 = add_text_box(slide1, Inches(1.2), Inches(5.3), Inches(6), Inches(1.5))
set_para(tf4, "📋  Subject: Image Processing", size=16, color=MUTED)
set_para(tf4, "📐  Architecture: MobileNetV3 + LBP Texture + YCrCb Colour Fusion", size=16, color=MUTED, space_before=6)
set_para(tf4, "📊  Metrics: EER, APCER, BPCER, HTER (illustrative)", size=16, color=MUTED, space_before=6)
set_para(tf4, "🛠️  Framework: PyTorch  |  Python 3.10+", size=16, color=MUTED, space_before=6)

# Right side — key stats cards
card_x = Inches(8.5)
card_w = Inches(3.8)
card_h = Inches(1.1)

# Card 1 - Accuracy
card1 = add_shape(slide1, card_x, Inches(1.8), card_w, card_h, BG_CARD, 0.05)
tf_c1 = card1.text_frame
tf_c1.word_wrap = True
p = tf_c1.paragraphs[0]
p.text = "Synthetic Demo Accuracy"
p.font.size = Pt(13)
p.font.color.rgb = MUTED
p.font.name = "Calibri"
p2 = tf_c1.add_paragraph()
p2.text = "100.00%"
p2.font.size = Pt(32)
p2.font.color.rgb = GREEN
p2.font.bold = True
p2.font.name = "Calibri"

# Card 2 - EER
card2 = add_shape(slide1, card_x, Inches(3.15), card_w, card_h, BG_CARD, 0.05)
tf_c2 = card2.text_frame
tf_c2.word_wrap = True
p = tf_c2.paragraphs[0]
p.text = "Synthetic Demo EER"
p.font.size = Pt(13)
p.font.color.rgb = MUTED
p.font.name = "Calibri"
p2 = tf_c2.add_paragraph()
p2.text = "0.00%"
p2.font.size = Pt(32)
p2.font.color.rgb = ACCENT
p2.font.bold = True
p2.font.name = "Calibri"

# Card 3 - Parameters
card3 = add_shape(slide1, card_x, Inches(4.5), card_w, card_h, BG_CARD, 0.05)
tf_c3 = card3.text_frame
tf_c3.word_wrap = True
p = tf_c3.paragraphs[0]
p.text = "⚙️  Model Parameters"
p.font.size = Pt(13)
p.font.color.rgb = MUTED
p.font.name = "Calibri"
p2 = tf_c3.add_paragraph()
p2.text = "1,640,418"
p2.font.size = Pt(32)
p2.font.color.rgb = ORANGE
p2.font.bold = True
p2.font.name = "Calibri"


# ─────────────────────────────────────────────────────────────────────
#  SLIDE 2 — Problem Statement & Why We Need It
# ─────────────────────────────────────────────────────────────────────
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide2, BG_DARK)
add_shape(slide2, Inches(0), Inches(0), SLIDE_W, Inches(0.06), ACCENT)

# Title
tf = add_text_box(slide2, Inches(0.8), Inches(0.4), Inches(12), Inches(0.8))
set_para(tf, "⚠️  The Problem: Why Face Anti-Spoofing?", size=32, color=WHITE, bold=True)

# Divider
add_shape(slide2, Inches(0.8), Inches(1.15), Inches(3.5), Inches(0.04), RED)

# LEFT COLUMN — The Threat
card_left = add_shape(slide2, Inches(0.5), Inches(1.6), Inches(5.8), Inches(5.5), BG_CARD, 0.03)

tf_left = add_text_box(slide2, Inches(0.8), Inches(1.8), Inches(5.3), Inches(5.0))
set_para(tf_left, "🚨  The Threat Landscape", size=22, color=RED, bold=True)

attacks = [
    ("📸  Print Attacks", "Printed photos of a victim held in front of camera"),
    ("📱  Replay Attacks", "Video/photo of victim displayed on a screen"),
    ("🎭  3D Mask Attacks", "Silicone or 3D-printed masks mimicking facial geometry"),
    ("💻  Deepfake Injection", "AI-generated synthetic face videos injected into the camera stream"),
]
for title, desc in attacks:
    add_bullet(tf_left, title, size=16, color=ACCENT, bold=True, space_before=12)
    add_bullet(tf_left, desc, size=13, color=LIGHT, indent_level=1, space_before=2)

add_bullet(tf_left, "", size=8, color=LIGHT, space_before=8)
add_bullet(tf_left, "💰  Global losses from facial recognition fraud exceed",
           size=14, color=ORANGE, bold=True, space_before=6)
add_bullet(tf_left, "     $3.4 Billion annually (2024 estimates)", size=14, color=ORANGE, bold=True)

# RIGHT COLUMN — Why We Need It
card_right = add_shape(slide2, Inches(6.7), Inches(1.6), Inches(6.1), Inches(5.5), BG_CARD, 0.03)

tf_right = add_text_box(slide2, Inches(7.0), Inches(1.8), Inches(5.5), Inches(5.0))
set_para(tf_right, "🛡️  Why We Need Anti-Spoofing", size=22, color=GREEN, bold=True)

reasons = [
    ("🏦  Banking & Finance", "UPI, mobile banking apps use face auth — must prevent unauthorized access"),
    ("🛂  Border Control & Immigration", "Airports use facial recognition for identity verification at e-gates"),
    ("📱  Device Unlock", "iPhone FaceID, Android face unlock need liveness verification"),
    ("🏥  Healthcare", "Patient identity verification for medical records and insurance"),
    ("🏢  Enterprise Security", "Access control to secure buildings, data centers, and classified areas"),
]
for title, desc in reasons:
    add_bullet(tf_right, title, size=16, color=ACCENT, bold=True, space_before=10)
    add_bullet(tf_right, desc, size=12, color=LIGHT, indent_level=1, space_before=2)


# ─────────────────────────────────────────────────────────────────────
#  SLIDE 3 — Architecture & How It Works
# ─────────────────────────────────────────────────────────────────────
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide3, BG_DARK)
add_shape(slide3, Inches(0), Inches(0), SLIDE_W, Inches(0.06), ACCENT2)

# Title
tf = add_text_box(slide3, Inches(0.8), Inches(0.3), Inches(12), Inches(0.8))
set_para(tf, "🧠  System Architecture: How It Works", size=32, color=WHITE, bold=True)
add_shape(slide3, Inches(0.8), Inches(1.05), Inches(3.5), Inches(0.04), ACCENT2)

# ── Three Branch Cards ──

# Branch 1 — RGB Deep Features
b1 = add_shape(slide3, Inches(0.4), Inches(1.5), Inches(3.9), Inches(3.0), BG_CARD, 0.03)
tf1 = add_text_box(slide3, Inches(0.6), Inches(1.65), Inches(3.5), Inches(2.7))
set_para(tf1, "🖼️  Branch 1: Deep Features", size=18, color=ACCENT, bold=True)
set_para(tf1, "INPUT", size=11, color=MUTED, space_before=10)
set_para(tf1, "RGB Image (3 × 128 × 128)", size=14, color=WHITE, space_before=2)
set_para(tf1, "BACKBONE", size=11, color=MUTED, space_before=8)
set_para(tf1, "MobileNetV3-Small (ImageNet pretrained)", size=14, color=WHITE, space_before=2)
set_para(tf1, "OUTPUT", size=11, color=MUTED, space_before=8)
set_para(tf1, "576-D feature vector", size=14, color=GREEN, bold=True, space_before=2)
set_para(tf1, "Captures high-level facial structure, shape, and semantic features", size=11, color=MUTED, space_before=8)

# Branch 2 — LBP Texture
b2 = add_shape(slide3, Inches(4.6), Inches(1.5), Inches(3.9), Inches(3.0), BG_CARD, 0.03)
tf2 = add_text_box(slide3, Inches(4.8), Inches(1.65), Inches(3.5), Inches(2.7))
set_para(tf2, "🔬  Branch 2: Texture (LBP)", size=18, color=ACCENT, bold=True)
set_para(tf2, "INPUT", size=11, color=MUTED, space_before=10)
set_para(tf2, "LBP Map (1 × 128 × 128)", size=14, color=WHITE, space_before=2)
set_para(tf2, "NETWORK", size=11, color=MUTED, space_before=8)
set_para(tf2, "4-layer Light CNN", size=14, color=WHITE, space_before=2)
set_para(tf2, "OUTPUT", size=11, color=MUTED, space_before=8)
set_para(tf2, "64-D feature vector", size=14, color=GREEN, bold=True, space_before=2)
set_para(tf2, "Detects micro-texture artifacts: moiré patterns, print dots, screen pixels", size=11, color=MUTED, space_before=8)

# Branch 3 — YCrCb Colour
b3 = add_shape(slide3, Inches(8.8), Inches(1.5), Inches(3.9), Inches(3.0), BG_CARD, 0.03)
tf3 = add_text_box(slide3, Inches(9.0), Inches(1.65), Inches(3.5), Inches(2.7))
set_para(tf3, "🎨  Branch 3: Colour (YCrCb)", size=18, color=ACCENT, bold=True)
set_para(tf3, "INPUT", size=11, color=MUTED, space_before=10)
set_para(tf3, "YCrCb Image (3 × 128 × 128)", size=14, color=WHITE, space_before=2)
set_para(tf3, "NETWORK", size=11, color=MUTED, space_before=8)
set_para(tf3, "4-layer Light CNN", size=14, color=WHITE, space_before=2)
set_para(tf3, "OUTPUT", size=11, color=MUTED, space_before=8)
set_para(tf3, "64-D feature vector", size=14, color=GREEN, bold=True, space_before=2)
set_para(tf3, "Skin chrominance differs between live skin vs paper/screen", size=11, color=MUTED, space_before=8)

# ── Fusion & Pipeline ──
# Arrow indicator
tf_arrow = add_text_box(slide3, Inches(0.4), Inches(4.65), Inches(12.5), Inches(0.5))
set_para(tf_arrow, "⬇️  ───────────────  CONCATENATE (576 + 64 + 64 = 704-D)  ───────────────  ⬇️",
         size=16, color=ACCENT2, bold=True, alignment=PP_ALIGN.CENTER)

# Fusion card
fusion = add_shape(slide3, Inches(0.4), Inches(5.2), Inches(12.5), Inches(2.0), BG_CARD, 0.03)
tf_f = add_text_box(slide3, Inches(0.7), Inches(5.3), Inches(12), Inches(1.8))
set_para(tf_f, "⚡  Fusion & Classification Pipeline", size=20, color=ACCENT2, bold=True)

# Pipeline flow
set_para(tf_f, "704-D Fused Vector  →  FC(256) + BN + ReLU + Dropout(0.4)  →  FC(128) + BN + ReLU + Dropout(0.2)  →  FC(2)  →  Softmax  →  [LIVE  |  SPOOF]",
         size=16, color=WHITE, space_before=10)

set_para(tf_f, "Training: AdamW optimizer  |  Cosine Annealing LR (1e-3 → 1e-6)  |  CrossEntropy Loss  |  15 Epochs  |  Kaiming Initialization",
         size=13, color=MUTED, space_before=10)


# ─────────────────────────────────────────────────────────────────────
#  SLIDE 4 — Results, Metrics & Conclusion
# ─────────────────────────────────────────────────────────────────────
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide4, BG_DARK)
add_shape(slide4, Inches(0), Inches(0), SLIDE_W, Inches(0.06), GREEN)

# Title
tf = add_text_box(slide4, Inches(0.8), Inches(0.3), Inches(12), Inches(0.8))
set_para(tf, "📊  Example Metrics & Results", size=32, color=WHITE, bold=True)
add_shape(slide4, Inches(0.8), Inches(1.05), Inches(3.5), Inches(0.04), GREEN)

# ── LEFT: Metrics Table ──
metrics_card = add_shape(slide4, Inches(0.4), Inches(1.5), Inches(6), Inches(3.5), BG_CARD, 0.03)
tf_m = add_text_box(slide4, Inches(0.7), Inches(1.6), Inches(5.5), Inches(3.3))
set_para(tf_m, "Example Metrics (Synthetic Holdout)", size=20, color=GREEN, bold=True)

metrics_data = [
    ("Equal Error Rate (EER)", "0.00%", GREEN),
    ("APCER (Synthetic Attack Samples)", "0.00%", GREEN),
    ("BPCER (Synthetic Bona Fide Samples)", "0.00%", GREEN),
    ("HTER (Half Total Error)", "0.00%", GREEN),
    ("Synthetic Test Accuracy", "100.00%", ACCENT),
    ("Training Loss (final)", "0.0011", ACCENT),
]
for metric, value, clr in metrics_data:
    p = tf_m.add_paragraph()
    p.text = f"  {metric}"
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT
    p.font.name = "Calibri"
    p.space_before = Pt(6)

    p2 = tf_m.add_paragraph()
    p2.text = f"      ➜  {value}"
    p2.font.size = Pt(16)
    p2.font.color.rgb = clr
    p2.font.bold = True
    p2.font.name = "Calibri"
    p2.space_before = Pt(1)

# ── RIGHT TOP: Confusion Matrix ──
cm_card = add_shape(slide4, Inches(6.8), Inches(1.5), Inches(5.8), Inches(1.8), BG_CARD, 0.03)
tf_cm = add_text_box(slide4, Inches(7.1), Inches(1.6), Inches(5.3), Inches(1.6))
set_para(tf_cm, "📋  Confusion Matrix", size=18, color=ACCENT, bold=True)
set_para(tf_cm, "  TP (Real → Real)     :  80    ✅", size=14, color=GREEN, space_before=8,
         font_name="Courier New")
set_para(tf_cm, "  TN (Spoof → Spoof)  :  80    ✅", size=14, color=GREEN, space_before=2,
         font_name="Courier New")
set_para(tf_cm, "  FP (Spoof → Real)    :   0    🟢", size=14, color=LIGHT, space_before=2,
         font_name="Courier New")
set_para(tf_cm, "  FN (Real → Spoof)    :   0    🟢", size=14, color=LIGHT, space_before=2,
         font_name="Courier New")

# ── RIGHT BOTTOM: Applications ──
app_card = add_shape(slide4, Inches(6.8), Inches(3.6), Inches(5.8), Inches(1.5), BG_CARD, 0.03)
tf_app = add_text_box(slide4, Inches(7.1), Inches(3.7), Inches(5.3), Inches(1.3))
set_para(tf_app, "Potential Use Cases (Not Validated)", size=18, color=ORANGE, bold=True)
apps = ["🏦 Banking & UPI Auth", "🛂 Airport e-Gates", "📱 Phone Face Unlock",
        "🏥 Healthcare ID", "🔒 Enterprise Access", "👮 Law Enforcement"]
set_para(tf_app, "   ".join(apps), size=12, color=LIGHT, space_before=8)

# ── BOTTOM: Tech Stack & Conclusion ──
bottom_card = add_shape(slide4, Inches(0.4), Inches(5.3), Inches(12.5), Inches(1.9), BG_CARD, 0.03)
tf_b = add_text_box(slide4, Inches(0.7), Inches(5.4), Inches(12), Inches(1.7))
set_para(tf_b, "🔧  Technology Stack & Key Takeaways", size=20, color=ACCENT2, bold=True)

set_para(tf_b, "Tech:  PyTorch  •  MobileNetV3  •  OpenCV  •  NumPy  •  scikit-learn  •  Matplotlib    |    "
               "Input: 128×128 RGB + LBP + YCrCb    |    Model Size: 6.7 MB (1.6M params)",
         size=13, color=MUTED, space_before=8)

set_para(tf_b, "✅ Multi-stream fusion combines RGB, texture, and color cues   "
               "✅ Lightweight MobileNetV3 backbone supports frame-wise inference   "
               "✅ Metric calculations are illustrative, not ISO/IEC certification   "
               "✅ Live webcam demo with Haar cascade face detection + per-face liveness scoring",
         size=13, color=GREEN, space_before=8)


# ══════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════
prs.save(SAVE_PATH)
print(f"\n✅  Presentation saved → {SAVE_PATH}")
print(f"    Slides: 4")
print(f"    Open it:  open \"{SAVE_PATH}\"\n")
