# TrustedText

> High-performance AI text detection using neural networks with contrastive learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

TrustedText is a robust AI text detection system that uses deep learning to distinguish between human-written and AI-generated content. Built with neural networks and contrastive learning, it provides accurate detection with support for NVIDIA GPUs, Apple Silicon, and CPU training.

## Why TrustedText?

- **State-of-the-art Detection**: Neural architecture with contrastive learning for robust AI text identification
- **Flexible Hardware Support**: Automatically detects and optimizes for NVIDIA CUDA, Apple MPS, or CPU
- **Production Ready**: Train once on powerful hardware, deploy anywhere
- **Easy to Use**: Simple API for both training and inference

## Quick Start

### Installation

```bash
# Install PyTorch (choose one based on your hardware)
pip install torch --index-url https://download.pytorch.org/whl/cu128  # NVIDIA RTX 50-series
pip install torch --index-url https://download.pytorch.org/whl/cu121  # NVIDIA RTX 40-series
pip install torch  # CPU or Apple Silicon

# Install dependencies
pip install sentence-transformers scikit-learn numpy
```

> **Note**: For RTX 50-series GPUs, see the [GPU Setup Guide](docs/GPU_SETUP.md) for important compatibility information.

### Train Your Model

```python
from trustedText import TrustedText

detector = TrustedText()
detector.run_full_pipeline()
```

### Detect AI Text

```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()

label, probability = detector.predict("Your text here")
print(f"Prediction: {label} (AI probability: {probability:.3f})")
```

## Features

### Core Capabilities

- **Contrastive Learning**: Learns discriminative representations by pulling same-class samples together
- **Hard Negative Mining**: Automatically focuses on difficult examples during training
- **Embeddings Cache**: Speeds up repeated training sessions
- **Cross-Validation**: Built-in evaluation metrics and test set support
- **Batch Processing**: Efficient inference on multiple texts

### Technical Highlights

- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B (768 dimensions)
- **Neural Architecture**: 2-layer encoder with LayerNorm and dropout
- **Training Strategy**: Supervised contrastive loss + cross-entropy with early stopping
- **Optimizer**: AdamW with cosine annealing learning rate schedule

## Documentation

| Guide | Description |
|-------|-------------|
| [Web Interface](docs/START.md) | Run the web UI for interactive detection |
| [GPU Setup](docs/GPU_SETUP.md) | Configure NVIDIA GPUs and troubleshoot CUDA |
| [Model Transfer](docs/TRANSFER_MODEL.md) | Train on remote GPU machines and transfer models |

## Training Data Setup

TrustedText requires you to provide your own training data. Organize your data as follows:

```
data/
├── human/              # Human-written text files
│   ├── text001.txt
│   ├── text002.txt
│   └── ...
├── ai/                 # AI-generated text files
│   ├── text001.txt
│   ├── text002.txt
│   └── ...
└── test/              # Optional test set
    ├── human/
    └── ai/
```

**Collection Tips**:
- Aim for 100+ samples per category
- Include diverse sources (news, blogs, academic, social media)
- Use UTF-8 plain text format
- One sample per file

## Usage Examples

### Basic Inference

```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()

# Single prediction
label, prob = detector.predict("Your text here")

# With custom threshold
label, prob = detector.predict(text, threshold=0.7)  # More conservative
```

### Batch Processing

```python
detector = TrustedText()
detector.load()

texts = ["Sample 1", "Sample 2", "Sample 3"]
results = [detector.predict(text) for text in texts]

for text, (label, prob) in zip(texts, results):
    print(f"{label}: {prob:.3f} - {text[:50]}...")
```

### Custom Configuration

```python
from trustedText import TrustedText, Config

config = Config(
    hidden_dim=512,
    dropout=0.4,
    epochs=100,
    learning_rate=1e-3,
    contrastive_weight=0.5,
    use_contrastive=True
)

detector = TrustedText(config=config)
detector.run_full_pipeline()
```

## Performance Benchmarks

Training time for 1000 samples:

| Hardware | Training Time |
|----------|---------------|
| NVIDIA RTX 4090 | ~30 seconds |
| Apple M3 Pro | ~60 seconds |
| Intel i7 CPU | ~180 seconds |

Model size: 100-500 MB (varies with training data size)

## Configuration

Control training behavior with environment variables:

```bash
export MAX_SEQ_LEN=320          # Maximum sequence length
export EMBED_BATCH_SIZE=4       # Batch size (adjust for GPU memory)
```

## Project Structure

```
trustedText/
├── trustedText.py          # Core library (training & inference)
├── main.py                 # CLI entry point
├── app/
│   ├── api.py             # FastAPI server
│   └── index.html         # Web interface
├── data/                  # Training data (not included)
├── models/                # Trained models (auto-generated)
└── docs/                  # Documentation
```

## Troubleshooting

### CUDA Not Available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

Reduce batch size:
```bash
export EMBED_BATCH_SIZE=2
python trustedText.py
```

### Model Not Found

Train a model first or transfer from another machine:
```bash
python trustedText.py  # Train locally
```

See [Model Transfer Guide](docs/TRANSFER_MODEL.md) for remote training.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TrustedText in your research or project, please cite:

```bibtex
@misc{trustedtext2026,
    title={TrustedText: AI Text Detection using Neural Networks with Contrastive Learning},
    author={Timmy Wu and Matthew Hung},
    year={2026},
    publisher={GitHub},
    howpublished={\url{https://github.com/theoncetimes/trustedText}}
}
```

## Support

- **Documentation**: Check the [docs/](docs/) folder
- **Issues**: [Open an issue](https://github.com/theoncetimes/trustedText/issues) on GitHub

---

Made with focus on accuracy, performance, and ease of use.
