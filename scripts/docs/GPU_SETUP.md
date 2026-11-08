# GPU Setup Guide for TrustedText

## Overview

This guide explains how to set up GPU acceleration for the TrustedText AI text detection system, with special focus on newer NVIDIA GPUs like the RTX 50-series (Blackwell architecture).

## GPU Compatibility

### Supported GPUs

- **RTX 40-series and older**: Fully supported with standard PyTorch installation (sm_89 and below)
- **RTX 50-series (5060, 5070, 5080, 5090)**: Requires PyTorch with CUDA 12.8+ support (sm_120)

### Compute Capability

Your GPU's compute capability determines PyTorch compatibility:

- **sm_50-sm_90**: Standard PyTorch releases
- **sm_120** (RTX 50-series): Requires PyTorch 2.10.0+ with CUDA 12.8+

## Installation Instructions

### 1. Check Your GPU

First, verify your GPU model and CUDA version:

```cmd
nvidia-smi
```

This will show:
- GPU model (e.g., "NVIDIA GeForce RTX 5060")
- Driver version
- CUDA version supported by your driver

### 2. Install PyTorch with Appropriate CUDA Version

#### For RTX 50-series (Blackwell - sm_120)

The RTX 50-series requires PyTorch with CUDA 12.8 or higher:

```cmd
# Using pip3 (recommended) - DO NOT install torchvision for text-only projects
.venv\Scripts\pip3.exe install torch --index-url https://download.pytorch.org/whl/cu128

# OR using regular pip
.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu128
```

**⚠️ Important**: Do NOT install `torchvision` - it causes conflicts with `transformers` and is not needed for text processing.

#### For RTX 40-series and Older

```cmd
# CUDA 12.1 (most compatible) - DO NOT install torchvision
.venv\Scripts\pip3.exe install torch --index-url https://download.pytorch.org/whl/cu121

# OR CUDA 11.8 (for older drivers)
.venv\Scripts\pip3.exe install torch --index-url https://download.pytorch.org/whl/cu118
```

**⚠️ Important**: Do NOT install `torchvision` - it causes conflicts with `transformers` and is not needed for text processing.

### 3. Verify Installation

Check if PyTorch detects your GPU:

```cmd
.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5060
```

## Troubleshooting

### Error: "CUDA error: no kernel image is available for execution on the device"

**Cause**: Your PyTorch version doesn't support your GPU's compute capability.

**Solution for RTX 50-series**:
1. Uninstall current PyTorch and torchvision:
   ```cmd
   uv pip uninstall torch torchvision
   ```

2. Install PyTorch with CUDA 12.8 (without torchvision):
   ```cmd
   .venv\Scripts\pip3.exe install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. Verify:
   ```cmd
   .venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); cap = torch.cuda.get_device_capability(0); print(f'Compute capability: sm_{cap[0]*10+cap[1]}')"
   ```

### Error: "ModuleNotFoundError: Could not import module 'PreTrainedModel'"

**Cause**: Broken or incompatible `transformers` installation.

**Solution**:
```cmd
uv pip uninstall transformers
uv pip install transformers
```

### CPU Fallback (If GPU Still Doesn't Work)

If you continue to have issues, the system will automatically fall back to CPU mode. While slower, it will still work correctly.

To force CPU mode:
```cmd
set CUDA_VISIBLE_DEVICES=-1
.venv\Scripts\python.exe train.py
```

## Performance Expectations

### With GPU (RTX 5060)
- Model loading: ~5-10 seconds
- Encoding 34 texts: ~5-15 seconds
- **Speed improvement**: 5-10x faster than CPU

### With CPU
- Model loading: ~20-40 seconds
- Encoding 34 texts: ~60-120 seconds
- Still functional, just slower

## Environment Variables

### Control Device Selection

```cmd
# Force CPU (disable GPU)
set CUDA_VISIBLE_DEVICES=-1

# Use specific GPU (if you have multiple)
set CUDA_VISIBLE_DEVICES=0

# Increase batch size for faster GPU processing
set EMBED_BATCH_SIZE=8

# Adjust sequence length
set MAX_SEQ_LEN=512
```

### Debugging GPU Issues

```cmd
# Enable verbose CUDA error reporting
set CUDA_LAUNCH_BLOCKING=1
.venv\Scripts\python.exe -m trustedtext.train
```

## Hardware Recommendations

### Minimum Requirements
- **GPU**: Any NVIDIA GPU with 4GB+ VRAM and compute capability sm_50+
- **RAM**: 8GB system RAM
- **Storage**: 5GB free space (for models and cache)

### Recommended Configuration
- **GPU**: RTX 3060 or better (8GB+ VRAM)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space (SSD preferred)

### Optimal Performance
- **GPU**: RTX 4070 or RTX 5060+ (12GB+ VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 20GB free space on NVMe SSD

## Common GPU-Related Issues

### 1. Out of Memory Errors

If you see CUDA out of memory errors:

```cmd
# Reduce batch size
set EMBED_BATCH_SIZE=2
.venv\Scripts\python.exe -m trustedtext.train
```

### 2. GPU Not Detected

Possible causes:
- Outdated NVIDIA drivers
- CUDA driver version too old
- GPU is being used by another process

**Solution**:
1. Update NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
2. Restart your computer
3. Close GPU-intensive applications (Chrome, games, etc.)
4. Try again

### 3. Slow Performance Despite GPU

Check GPU utilization:
```cmd
nvidia-smi
```

If GPU utilization is low:
- Increase batch size: `set EMBED_BATCH_SIZE=16`
- Increase sequence length: `set MAX_SEQ_LEN=512`
- Check if thermal throttling is occurring (temperature too high)

## Version Compatibility Matrix

| GPU Architecture | Compute Capability | Min PyTorch | Recommended CUDA |
|-----------------|-------------------|-------------|------------------|
| Maxwell (GTX 900) | sm_50-52 | 2.0.0+ | 11.8 |
| Pascal (GTX 10xx) | sm_60-61 | 2.0.0+ | 11.8 |
| Volta (Titan V) | sm_70 | 2.0.0+ | 11.8 |
| Turing (RTX 20xx) | sm_75 | 2.0.0+ | 11.8/12.1 |
| Ampere (RTX 30xx) | sm_80-86 | 2.0.0+ | 11.8/12.1 |
| Ada (RTX 40xx) | sm_89 | 2.0.0+ | 12.1/12.4 |
| **Blackwell (RTX 50xx)** | **sm_120** | **2.10.0+** | **12.8+** |

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Compute Capability Reference](https://developer.nvidia.com/cuda-gpus)

## Getting Help

If you continue to experience issues:

1. Check the error message carefully
2. Verify your GPU is detected: `nvidia-smi`
3. Check PyTorch version: `.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"`
4. Try CPU fallback mode as a temporary solution
5. Open an issue with:
   - GPU model
   - Driver version
   - PyTorch version
   - Full error message

## Summary

**For RTX 50-series users (5060, 5070, 5080, 5090)**:
```cmd
# One-time setup (DO NOT install torchvision)
uv pip uninstall torch torchvision
.venv\Scripts\pip3.exe install torch --index-url https://download.pytorch.org/whl/cu128

# Verify
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run training
.venv\Scripts\python.exe -m trustedtext.train
```

Your system should now leverage GPU acceleration for significantly faster training and inference! 🚀
