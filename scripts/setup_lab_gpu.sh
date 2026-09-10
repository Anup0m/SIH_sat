#!/usr/bin/env bash
# SatQuery AI — Linux / College Lab GPU Environment Setup
set -e

echo "=========================================================="
echo "🛰️  SatQuery AI — College Lab GPU Environment Provisioning"
echo "=========================================================="

echo "[*] Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "[!] Warning: nvidia-smi not found. CUDA acceleration may not be active."
fi

echo "[*] Creating virtual environment..."
python3 -m venv venv_satquery
source venv_satquery/bin/activate

echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[*] Installing SatQuery dependencies..."
pip install -r requirements.txt

echo "[*] Running Hardware Diagnostics..."
python training/launcher.py --task eval

echo "=========================================================="
echo "✅ Lab Environment Ready!"
echo "To start training:"
echo "  1. source venv_satquery/bin/activate"
echo "  2. python training/launcher.py --task all"
echo "=========================================================="
