"""
Upload the TrustedText model to Hugging Face Hub.

This script packages and uploads:
- The trained KNN classifier (knn_classifier.pkl)
- Model card with usage instructions
- Example inference code
- Requirements file

Prerequisites:
1. Install huggingface_hub: pip install huggingface_hub
2. Login to HuggingFace: huggingface-cli login
3. Train your model: python -m trustedtext.train
"""

import pickle
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import json
import os

# Configuration
REPO_NAME = "trustedtext-ai-detector"  # Change this to your desired repo name
USERNAME = None  # Will use your logged-in username if None
REPO_TYPE = "model"
PRIVATE = False  # Set to True for private repo

def create_model_card():
    """Create a comprehensive model card in README.md"""
    
    # Load model info
    # models directory is at project root, not in scripts/
    model_path = Path(__file__).resolve().parent.parent / "models" / "knn_classifier.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Please run 'python -m trustedtext.train' first.")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    human_count = model_data.get("human_count", 0)
    ai_count = model_data.get("ai_count", 0)
    model_id = model_data.get("model_id", "Qwen/Qwen3-Embedding-0.6B")
    max_seq_length = model_data.get("max_seq_length", 320)
    
    model_card = f"""---
tags:
- text-classification
- ai-detection
- knn
- sentence-transformers
library_name: sentence-transformers
pipeline_tag: text-classification
license: mit
datasets:
- custom
language:
- en
metrics:
- accuracy
- precision
- recall
---

# TrustedText - AI Text Detection Model

An AI text detection system using embedding-based classification with K-Nearest Neighbors (KNN).

## Model Description

This model detects whether text is AI-generated or human-written using:
- **Base Model**: [{model_id}]({model_id})
- **Classifier**: K-Nearest Neighbors (KNN) with cosine similarity
- **Training Data**: {human_count} human-written samples, {ai_count} AI-generated samples
- **Max Sequence Length**: {max_seq_length} tokens

## How It Works

1. Text is encoded using the Qwen3-Embedding model
2. The embedding is compared to cached training embeddings
3. KNN classifier predicts based on nearest neighbors
4. Returns probability score and label (Human/AI)

## Installation

```bash
pip install sentence-transformers torch numpy scikit-learn huggingface_hub
```

## Usage

### Quick Start

```python
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="theoncetimes/{REPO_NAME}",
    filename="knn_classifier.pkl"
)

# Load classifier
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

knn = model_data["knn"]
X_train = model_data["X_train"]
instruction = model_data["instruction"]
MODEL_ID = model_data["model_id"]
max_seq_length = model_data["max_seq_length"]

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(MODEL_ID, device=device, trust_remote_code=True)
embedding_model.max_seq_length = max_seq_length

# Detect function
def detect_ai_text(text: str, k: int = 3, threshold: float = 0.5):
    prompt = instruction + text
    embedding = embedding_model.encode(
        [prompt],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    proba = knn.predict_proba(embedding)[0]
    ai_probability = float(proba[1])
    label = "AI" if ai_probability >= threshold else "Human"
    
    return {{
        "label": label,
        "ai_probability": ai_probability,
        "human_probability": float(proba[0])
    }}

# Example usage
result = detect_ai_text("This is a sample text to analyze.")
print(f"Label: {{result['label']}}")
print(f"AI Probability: {{result['ai_probability']:.3f}}")
```

### Using with transformers pipeline (custom implementation)

```python
from huggingface_hub import hf_hub_download
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class TrustedTextDetector:
    def __init__(self, repo_id: str):
        # Download and load model
        model_path = hf_hub_download(repo_id=repo_id, filename="knn_classifier.pkl")
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.knn = model_data["knn"]
        self.instruction = model_data["instruction"]
        self.max_seq_length = model_data["max_seq_length"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            model_data["model_id"],
            device=device,
            trust_remote_code=True
        )
        self.embedding_model.max_seq_length = self.max_seq_length
    
    def __call__(self, text: str, k: int = 3, threshold: float = 0.5):
        prompt = self.instruction + text
        embedding = self.embedding_model.encode(
            [prompt],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Get k nearest neighbors
        self.knn.n_neighbors = min(k, len(self.knn._fit_X))
        proba = self.knn.predict_proba(embedding)[0]
        
        return {{
            "label": "AI" if proba[1] >= threshold else "Human",
            "ai_probability": float(proba[1]),
            "human_probability": float(proba[0])
        }}

# Usage
detector = TrustedTextDetector("theoncetimes/{REPO_NAME}")
result = detector("Sample text to analyze")
print(result)
```

## API Parameters

- **k** (int, default=3): Number of nearest neighbors to consider
- **threshold** (float, default=0.5): Probability threshold for AI classification
  - Lower values (0.3) = more sensitive to AI text
  - Higher values (0.7) = require stronger evidence

## Model Performance

Training set size: {human_count + ai_count} samples
- Human samples: {human_count}
- AI samples: {ai_count}

Cross-validation and test metrics are computed during training. See training logs for detailed performance metrics.

## Training Data

The model was trained on:
- **Human-written text**: News articles, forum posts, and other authentic human content
- **AI-generated text**: Text from various LLMs including GPT, Claude, Gemini, DeepSeek, etc.

## Limitations

- Performance depends on text length and quality
- May struggle with very short texts (<50 words)
- Trained on English text primarily
- Performance may vary on domain-specific text
- Should not be used as the sole arbiter of AI detection

## Citation

```bibtex
@software{{trustedtext2026,
  title = {{TrustedText: AI Text Detection using KNN and Embeddings}},
  author = {{TrustedText Contributors}},
  year = {{2026}},
  url = {{https://huggingface.co/theoncetimes/{REPO_NAME}}}
}}
```

## License

MIT License

## Repository

Full source code and training scripts: [GitHub Repository]
(Add your GitHub link here)
"""
    
    return model_card

def create_requirements_file():
    """Create requirements.txt for the model"""
    requirements = """sentence-transformers>=5.2.0
torch>=2.10.0
numpy>=2.4.1
scikit-learn>=1.8.0
huggingface_hub>=0.20.0
"""
    return requirements

def create_inference_example():
    """Create a standalone inference example"""
    example = """import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import torch

# Configuration
REPO_ID = "theoncetimes/trustedtext-ai-detector"  # Change this to your org name

# Download model
print("Downloading model...")
model_path = hf_hub_download(repo_id=REPO_ID, filename="knn_classifier.pkl")

# Load classifier
print("Loading classifier...")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

knn = model_data["knn"]
instruction = model_data["instruction"]
MODEL_ID = model_data["model_id"]
max_seq_length = model_data["max_seq_length"]

# Load embedding model
print(f"Loading embedding model: {MODEL_ID}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embedding_model = SentenceTransformer(MODEL_ID, device=device, trust_remote_code=True)
embedding_model.max_seq_length = max_seq_length

print("Model loaded successfully!\\n")

# Detection function
def detect(text: str, k: int = 3, threshold: float = 0.5):
    prompt = instruction + text
    embedding = embedding_model.encode(
        [prompt],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # Adjust k if needed
    knn.n_neighbors = min(k, len(knn._fit_X))
    proba = knn.predict_proba(embedding)[0]
    ai_probability = float(proba[1])
    label = "AI" if ai_probability >= threshold else "Human"
    
    return label, ai_probability

# Example usage
if __name__ == "__main__":
    # Test samples
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a significant development, researchers have unveiled a groundbreaking approach...",
    ]
    
    for text in test_texts:
        label, prob = detect(text)
        print(f"Text: {text[:60]}...")
        print(f"Prediction: {label} (AI probability: {prob:.3f})\\n")
"""
    return example

def upload_to_hf():
    """Upload model and files to Hugging Face Hub"""
    
    # Check if model exists
    # models directory is at project root, not in scripts/
    model_path = Path(__file__).resolve().parent.parent / "models" / "knn_classifier.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Please run 'python -m trustedtext.train' first to train the model."
        )
    
    # Initialize API
    api = HfApi()
    
    # Get username
    username = USERNAME or api.whoami()["name"]
    repo_id = f"{username}/{REPO_NAME}"
    
    print(f"🚀 Uploading model to: {repo_id}")
    print(f"   Private: {PRIVATE}")
    
    # Create repository
    try:
        print("\\n📦 Creating repository...")
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            private=PRIVATE,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Create temporary directory for files
    temp_dir = Path("temp_hf_upload")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create README.md (model card)
        print("\\n📝 Creating model card...")
        readme_path = temp_dir / "README.md"
        readme_path.write_text(create_model_card(), encoding="utf-8")
        
        # Create requirements.txt
        print("📝 Creating requirements.txt...")
        req_path = temp_dir / "requirements.txt"
        req_path.write_text(create_requirements_file(), encoding="utf-8")
        
        # Create inference example
        print("📝 Creating inference example...")
        example_path = temp_dir / "inference_example.py"
        example_path.write_text(create_inference_example(), encoding="utf-8")
        
        # Upload model file
        print("\\n⬆️  Uploading model file...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="knn_classifier.pkl",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print("✅ Model file uploaded")
        
        # Upload README
        print("⬆️  Uploading model card...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print("✅ Model card uploaded")
        
        # Upload requirements
        print("⬆️  Uploading requirements...")
        api.upload_file(
            path_or_fileobj=str(req_path),
            path_in_repo="requirements.txt",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print("✅ Requirements uploaded")
        
        # Upload example
        print("⬆️  Uploading inference example...")
        api.upload_file(
            path_or_fileobj=str(example_path),
            path_in_repo="inference_example.py",
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print("✅ Inference example uploaded")
        
        print(f"\\n🎉 Upload complete!")
        print(f"\\n🔗 View your model at: https://huggingface.co/{repo_id}")
        print(f"\\n📥 Others can use it with:")
        print(f"    from huggingface_hub import hf_hub_download")
        print(f"    model = hf_hub_download('{repo_id}', 'knn_classifier.pkl')")
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\\n🧹 Cleaned up temporary files")

if __name__ == "__main__":
    print("=" * 60)
    print("TrustedText - Upload to Hugging Face Hub")
    print("=" * 60)
    
    try:
        upload_to_hf()
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Install huggingface_hub: pip install huggingface_hub")
        print("3. Train your model first: python -m trustedtext.train")
        raise
