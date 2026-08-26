#!/bin/bash
set -e

echo "=== Step 1: Creating Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Step 2: Installing OpenCV and PyTorch ==="
pip install opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 3: Pushing to GitHub ==="
git init
git remote add origin https://github.com/HarshadaGhanwat06/audio8.git
git add -A
git commit -m "Add venv setup and opencv"
git branch -M main
git push -u origin main --force

echo "=== Done ==="
