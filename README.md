# TrustedText - AI Text Detection

A high-performance AI text detection system using neural networks with contrastive learning. Supports training on NVIDIA GPUs, Apple Silicon (MPS), or CPU.

## Features

- **Neural Architecture**: Deep learning classifier with contrastive learning for robust detection
- **Multi-GPU Support**: Automatically detects and uses NVIDIA CUDA, Apple MPS, or CPU
- **Transfer Learning**: Train on powerful GPU machines and transfer models easily
- **Hard Negative Mining**: Automatically identifies and emphasizes difficult examples during training
- **Contrastive Learning**: Learns discriminative representations by pulling same-class samples together
- **Embeddings Cache**: Speeds up repeated training with cached embeddings
- **Cross-Validation**: Built-in evaluation metrics and test set support

## Quick Start

### 1. Installation

**⚠️ IMPORTANT for RTX 50-series GPUs (5060, 5070, 5080, 5090):**  
See [GPU_SETUP.md](docs/GPU_SETUP.md) for detailed GPU setup instructions.

```bash
# For NVIDIA RTX 50-series (Blackwell - CUDA 12.8+)
# DO NOT install torchvision - it causes conflicts!
pip install torch --index-url https://download.pytorch.org/whl/cu128

# For NVIDIA RTX 40-series and older (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU or Apple Silicon
pip install torch

# Install other dependencies
pip install sentence-transformers scikit-learn numpy
```

📖 **Having GPU issues?** Check the comprehensive [GPU Setup Guide](docs/GPU_SETUP.md)

### 2. Prepare Data

Organize your training data (you'll need to provide your own training data):

```
data/
├── human/          # Human-written texts
│   ├── text001.txt
│   ├── text002.txt
│   └── ...
├── ai/             # AI-generated texts
│   ├── text001.txt
│   ├── text002.txt
│   └── ...
└── test/           # Optional test set
    ├── human/
    └── ai/
```

**Note**: Training data is not included in this repository. You need to collect your own datasets of human-written and AI-generated texts.

### 3. Train Model

```bash
python trustedText.py
```

Or use the module programmatically:

```python
from trustedText import TrustedText

detector = TrustedText()
detector.run_full_pipeline()
```

Output:
```
NVIDIA GPU detected: [GPU name]
Using device: CUDA
Model loaded in 15.3 seconds | Dim: 768
...
Model saved to models/contrastive_classifier.pt
```

### 4. Use Model

**Option 1: Using the module**
```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()  # Load trained model
label, probability = detector.predict("Your text here")
print(f"Prediction: {label} (AI probability: {probability:.3f})")
```

**Option 2: Direct prediction**
```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()
result = detector.analyze_text("Your text here")
print(result)
```

## Training on Remote GPU Machine

See [TRANSFER_MODEL.md](docs/TRANSFER_MODEL.md) for detailed instructions on:
- Transferring data and code to GPU machines
- Manual SCP/rsync transfer
- Cloud storage options (S3, GCS, etc.)
- Downloading trained models
- Troubleshooting and optimization

## Project Structure

```
trustedText/
├── trustedText.py        # Main library module (training & inference)
├── main.py              # Simple entry point
├── app/                 # Web interface
│   ├── api.py          # FastAPI server
│   └── index.html      # Web UI
├── data/               # Training and test data (not included)
│   ├── human/         # Human-written texts
│   ├── ai/            # AI-generated texts
│   └── test/          # Test set
│       ├── human/
│       └── ai/
├── models/            # Trained models (auto-generated, not included)
├── docs/              # Documentation
│   ├── START.md       # Web interface usage guide
│   ├── GPU_SETUP.md   # GPU setup instructions
│   └── TRANSFER_MODEL.md # Model transfer guide
└── pyproject.toml     # Project dependencies
```

## Configuration

Environment variables:

```bash
# Maximum sequence length for embeddings
export MAX_SEQ_LEN=320

# Batch size for encoding (adjust based on GPU memory)
export EMBED_BATCH_SIZE=4
```

## Model Details

- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B (768 dimensions)
- **Classifier**: Neural network with contrastive learning
  - 2-layer encoder with LayerNorm and dropout
  - Projection head for contrastive embeddings
  - Classification head for binary prediction
- **Training Approach**: 
  - Supervised contrastive loss + cross-entropy loss
  - Hard negative mining to focus on difficult examples
  - Early stopping with patience
  - AdamW optimizer with cosine annealing
- **Training Data**: All provided samples (no train/test split)
- **Evaluation**: Separate test set from `data/test/`

## Performance

Training time (1000 samples):
- NVIDIA RTX 4090: ~30 seconds
- Apple M3 Pro: ~60 seconds
- CPU (Intel i7): ~180 seconds

Model size: ~100-500 MB (depending on training data size)

## Advanced Usage

### Custom Detection Threshold

```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()

# More conservative (fewer false positives)
label, prob = detector.predict(text, threshold=0.7)

# More aggressive (fewer false negatives)
label, prob = detector.predict(text, threshold=0.3)
```

### Batch Inference

```python
from trustedText import TrustedText

detector = TrustedText()
detector.load()

texts = ["Text 1", "Text 2", "Text 3"]
results = [detector.predict(text) for text in texts]

for text, (label, prob) in zip(texts, results):
    print(f"{label}: {prob:.3f} - {text[:50]}...")
```

### Model Configuration

```python
from trustedText import TrustedText, Config

# Customize model hyperparameters
config = Config(
    hidden_dim=512,        # Neural network hidden layer size
    dropout=0.4,           # Dropout rate for regularization
    epochs=100,            # Maximum training epochs
    learning_rate=1e-3,    # Learning rate
    contrastive_weight=0.5,  # Weight for contrastive loss
    use_contrastive=True   # Enable contrastive learning
)

detector = TrustedText(config=config)
```

## Troubleshooting

### CUDA Not Available

Check PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Out of Memory

Reduce batch size:
```bash
export EMBED_BATCH_SIZE=2
python train.py
```

### Model File Not Found

Run training first:
```bash
python trustedText.py
```

Or transfer from a GPU machine (see [TRANSFER_MODEL.md](docs/TRANSFER_MODEL.md)).

## Data Collection

This repository does not include training data. You need to collect your own datasets.

### Manual Collection

1. Create text files in `data/human/` for human-written content
2. Create text files in `data/ai/` for AI-generated content
3. Optionally add test samples to `data/test/human/` and `data/test/ai/`

**Tips:**
- Collect diverse samples (news, blogs, academic, social media)
- Aim for 100+ samples per category
- Each file should contain one text sample
- UTF-8 plain text format

See `data/README.md` for detailed guidelines.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues and questions, see the documentation in the `docs/` folder:
- [Web Interface Guide](docs/START.md) - Running the web UI
- [GPU Setup Guide](docs/GPU_SETUP.md) - GPU configuration
- [Model Transfer Guide](docs/TRANSFER_MODEL.md) - Transferring models between machines

Or open an issue on GitHub.



## Citation

If you use this codebase or otherwise find our work valuable, please cite:

@misc{trustedtext2026,
      title={TrustedText: AI Text Detection using Neural Networks with Contrastive Learning},
      author={TrustedText Contributors},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/theoncetimes/trustedText}},
}