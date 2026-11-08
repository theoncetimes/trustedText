#!/bin/bash
# Quick script to download trained model from GPU machine

set -e

# Configuration
GPU_HOST="${GPU_HOST:-user@gpu-machine}"
GPU_PATH="${GPU_PATH:-~/trustedText}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Project root is one level up from scripts/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================"
echo "Model Download Script"
echo "================================"
echo ""
echo "GPU Host: $GPU_HOST"
echo "GPU Path: $GPU_PATH"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if models directory exists on remote
echo "Checking remote models directory..."
if ssh "$GPU_HOST" "[ -d $GPU_PATH/models ]"; then
    echo "✓ Remote models directory found"
else
    echo "✗ Remote models directory not found at $GPU_PATH/models"
    echo "Have you run 'python -m trustedtext.train' on the GPU machine?"
    exit 1
fi

# Check if model file exists on remote
echo "Checking for model file..."
if ssh "$GPU_HOST" "[ -f $GPU_PATH/models/knn_classifier.pkl ]"; then
    echo "✓ Model file found"
    
    # Get file size
    SIZE=$(ssh "$GPU_HOST" "du -h $GPU_PATH/models/knn_classifier.pkl | cut -f1")
    echo "  Model size: $SIZE"
else
    echo "✗ Model file not found"
    exit 1
fi

# Download the model
echo ""
echo "Downloading model..."
mkdir -p "$PROJECT_ROOT/models"

rsync -avz --progress "$GPU_HOST:$GPU_PATH/models/" "$PROJECT_ROOT/models/"

echo ""
echo "================================"
echo "✓ Download complete!"
echo "================================"
echo ""
echo "Model saved to: $PROJECT_ROOT/models/"
echo ""
echo "To use the model, run:"
echo "  python scripts/load_model.py"
echo "  # or"
echo "  python -c \"from trustedText import TrustedText; d = TrustedText(); d.load(); print(d.predict('test'))\""
echo ""
