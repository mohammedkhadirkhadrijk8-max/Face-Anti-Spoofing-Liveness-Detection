#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Face Anti-Spoofing — Live Demo Launcher
#  Double-click this file or run:  ./run_live_demo.command
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🔐  Face Anti-Spoofing — Live Webcam Demo        ║"
echo "║   Press 'q' in the video window to quit             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

python3 inference.py

echo ""
echo "Demo finished. Press Enter to close."
read
