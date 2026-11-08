"""
Load and use a trained model for inference.
This script can load a model trained on another machine.
"""
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# Device detection
if torch.cuda.is_available():
    device = "cuda"
    print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Apple GPU (MPS) detected")
else:
    device = "cpu"
    print("No GPU detected, using CPU")

print(f"Using device: {device.upper()}\n")

# Load the trained model
# models directory is at project root, not in scripts/
model_dir = Path(__file__).resolve().parent.parent / "models"
model_path = model_dir / "knn_classifier.pkl"

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found at {model_path}\n"
        "Please run 'python -m trustedtext.train' first or transfer the model from another machine."
    )

print(f"Loading model from {model_path}...")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

knn = model_data["knn"]
X_train = model_data["X_train"]
y_train = model_data["y_train"]
instruction = model_data["instruction"]
MODEL_ID = model_data["model_id"]
max_seq_length = model_data["max_seq_length"]
human_count = model_data["human_count"]
ai_count = model_data["ai_count"]

print(f"Model loaded successfully!")
print(f"  Model ID: {MODEL_ID}")
print(f"  Max sequence length: {max_seq_length}")
print(f"  Training samples: {len(X_train)} (Human={human_count}, AI={ai_count})")
print(f"  KNN neighbors: {knn.n_neighbors}\n")

# Load the sentence transformer model
print(f"Loading sentence transformer model...")
model = SentenceTransformer(
    MODEL_ID,
    device=device,
    trust_remote_code=True
)
model.max_seq_length = max_seq_length
print(f"Sentence transformer loaded\n")

# Inference function
def detect(text: str, k=3, threshold=0.5, return_prob=True):
    """Detect if text is AI-generated.
    
    Args:
        text: Input text
        k: Number of neighbors
        threshold: Probability threshold for AI classification
        return_prob: Whether to return probability
    
    Returns:
        If return_prob=True: (label, probability)
        If return_prob=False: label only
    """
    prompt_text = instruction + text
    emb = model.encode([prompt_text], normalize_embeddings=True)
    
    knn.n_neighbors = min(k, len(X_train))
    probs = knn.predict_proba(emb)[0]
    ai_prob = probs[1] if len(probs) > 1 else 0.0
    
    label = "AI" if ai_prob >= threshold else "Human"
    
    if return_prob:
        return label, ai_prob
    return label


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing the loaded model")
    print("="*60)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In accordance with the aforementioned stipulations, it is imperative to underscore the significance of adhering to established protocols.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        label, prob = detect(text)
        print(f"\nTest {i}:")
        print(f"Text: {text[:80]}...")
        print(f"Prediction: {label} (AI probability: {prob:.3f})")
    
    print("\n" + "="*60)
    print("Model ready for inference!")
    print("Use: label, prob = detect('your text here')")
    print("="*60)
