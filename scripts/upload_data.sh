#!/bin/bash
# Quick script to upload training data to GPU machine

set -e

# Configuration
GPU_HOST="${GPU_HOST:-user@gpu-machine}"
GPU_PATH="${GPU_PATH:-~/trustedText}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Project root is one level up from scripts/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DIR="$PROJECT_ROOT"

echo "================================"
echo "Data Upload Script"
echo "================================"
echo ""
echo "GPU Host: $GPU_HOST"
echo "GPU Path: $GPU_PATH"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if data directory exists locally
if [ ! -d "$PROJECT_ROOT/data" ]; then
    echo "✗ Local data directory not found at $PROJECT_ROOT/data"
    exit 1
fi

echo "✓ Local data directory found"

# Count files
HUMAN_FILES=$(find "$PROJECT_ROOT/data/human" -name "*.txt" 2>/dev/null | wc -l)
AI_FILES=$(find "$PROJECT_ROOT/data/ai" -name "*.txt" 2>/dev/null | wc -l)
echo "  Human texts: $HUMAN_FILES files"
echo "  AI texts: $AI_FILES files"

# Create remote directory if needed
echo ""
echo "Setting up remote directory..."
ssh "$GPU_HOST" "mkdir -p $GPU_PATH"

# Upload data
echo ""
echo "Uploading data..."
rsync -avz --progress "$PROJECT_ROOT/data/" "$GPU_HOST:$GPU_PATH/data/"

# Upload training script
echo ""
echo "Uploading training script..."
rsync -avz --progress "$PROJECT_ROOT/src" "$PROJECT_ROOT/configs" \
    "$PROJECT_ROOT/pyproject.toml" "$PROJECT_ROOT/uv.lock" "$PROJECT_ROOT/README.md" \
    "$GPU_HOST:$GPU_PATH/"

echo ""
echo "================================"
echo "✓ Upload complete!"
echo "================================"
echo ""
echo "To train on GPU machine, run:"
echo "  ssh $GPU_HOST"
echo "  cd $GPU_PATH"
echo "  uv sync && uv run trustedtext-train"
echo ""
echo "After training, download the model with:"
echo "  ./scripts/download_model.sh"
echo ""
