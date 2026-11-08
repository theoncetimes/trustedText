"""
Quick script to verify your environment is set up correctly.
Run this before training to catch any issues early.
"""
import sys

print("="*60)
print("Environment Setup Check")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version}")
required_version = (3, 13)
if sys.version_info < required_version:
    print(f"⚠️  Warning: Python {required_version[0]}.{required_version[1]}+ recommended")
else:
    print("✓ Python version OK")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: {torch.__version__}")
    
    # Check device availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print("✓ Apple MPS (GPU) available")
    else:
        print("ℹ️  Using CPU (no GPU detected)")
        print("  For NVIDIA GPU support, run:")
        print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
except ImportError:
    print("\n✗ PyTorch not installed")
    print("  Run: uv sync")
    sys.exit(1)

# Check sentence-transformers
try:
    import sentence_transformers
    print(f"✓ sentence-transformers installed: {sentence_transformers.__version__}")
except ImportError:
    print("✗ sentence-transformers not installed")
    print("  Run: uv sync")
    sys.exit(1)

# Check scikit-learn
try:
    import sklearn
    print(f"✓ scikit-learn installed: {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn not installed")
    print("  Run: uv sync")
    sys.exit(1)

# Check numpy
try:
    import numpy as np
    print(f"✓ numpy installed: {np.__version__}")
except ImportError:
    print("✗ numpy not installed")
    print("  Run: uv sync")
    sys.exit(1)

# Check data directory
from pathlib import Path
# data directory is at project root, not in scripts/
data_dir = Path(__file__).resolve().parent.parent / "data"
if not data_dir.exists():
    print(f"\n✗ Data directory not found: {data_dir}")
    sys.exit(1)

human_dir = data_dir / "human"
ai_dir = data_dir / "ai"

if not human_dir.exists() or not ai_dir.exists():
    print(f"\n✗ Missing data folders")
    print(f"  Expected: {human_dir} and {ai_dir}")
    sys.exit(1)

human_files = list(human_dir.glob("*.txt"))
ai_files = list(ai_dir.glob("*.txt"))

print(f"\n✓ Data directory found: {data_dir}")
print(f"  Human texts: {len(human_files)} files")
print(f"  AI texts: {len(ai_files)} files")

if len(human_files) < 2 or len(ai_files) < 2:
    print("  ⚠️  Warning: Very few samples. Consider adding more data for better results.")
else:
    print("  ✓ Sufficient samples for training")

# Check test directory
test_dir = data_dir / "test"
if test_dir.exists():
    test_human = list((test_dir / "human").glob("*.txt")) if (test_dir / "human").exists() else []
    test_ai = list((test_dir / "ai").glob("*.txt")) if (test_dir / "ai").exists() else []
    print(f"\n✓ Test directory found")
    print(f"  Test human: {len(test_human)} files")
    print(f"  Test AI: {len(test_ai)} files")
else:
    print(f"\nℹ️  No test directory (optional)")

print("\n" + "="*60)
print("✓ Setup complete! Ready to train.")
print("="*60)
print("\nRun: python -m trustedtext.train")
print("="*60)
