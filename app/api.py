import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize FastAPI
app = FastAPI(title="TrustedText API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
classifier_model = None
scaler_mean = None
scaler_std = None
instruction = None
hyperparams = {}
metadata = {}
use_stylometric = True

# Request/Response models
class TextRequest(BaseModel):
    text: str

class DetectionResponse(BaseModel):
    label: str
    ai_probability: float
    confidence: str
    processing_time_ms: float

class StylometricFeatures:
    """Extract writing style features"""
    
    @staticmethod
    def extract(texts):
        features = []
        for text in texts:
            feat = []
            words = text.split()
            sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
            
            # 1. Text length features
            feat.append(len(words))
            feat.append(len(text))
            
            # 2. Sentence length statistics
            sent_lengths = [len(s.split()) for s in sentences] if sentences else [0]
            feat.append(np.mean(sent_lengths))
            feat.append(np.std(sent_lengths) if len(sent_lengths) > 1 else 0)
            
            # 3. Lexical diversity
            unique_words = set(w.lower() for w in words)
            lexical_div = len(unique_words) / max(len(words), 1)
            feat.append(lexical_div)
            
            # 4. Character-level diversity
            chars = list(text)
            unique_chars = set(chars)
            char_div = len(unique_chars) / max(len(chars), 1)
            feat.append(char_div)
            
            # 5. Punctuation patterns (match training exactly)
            punct_counts = {
                ',': text.count(','),
                ';': text.count(';'),
                ':': text.count(':'),
                '-': text.count('-'),
                '"': text.count('"') + text.count('"'),
                "'": text.count("'"),
            }
            total_punct = sum(punct_counts.values())
            feat.append(total_punct / max(len(words), 1))
            
            # 6. Function word density
            function_words = ['the', 'and', 'that', 'this', 'with', 'for', 'as', 'to', 'of', 'in']
            func_count = sum(1 for w in words if w.lower() in function_words)
            feat.append(func_count / max(len(words), 1))
            
            # 7. Average word length
            word_lengths = [len(w) for w in words]
            feat.append(np.mean(word_lengths) if word_lengths else 0)
            feat.append(np.std(word_lengths) if len(word_lengths) > 1 else 0)
            
            # 8. Paragraph structure (MISSING IN API!)
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            feat.append(len(paragraphs))
            
            features.append(feat)
        return np.array(features, dtype=np.float32)

class ContrastiveClassifier(torch.nn.Module):
    """Neural network for classification with optional projection layer"""
    
    def __init__(self, input_dim, hidden_dim=512, dropout=0.4, projection_dim=None):
        super().__init__()
        
        # Projection should process encoder output, not raw input
        if projection_dim is not None:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, projection_dim),  # FIX: hidden_dim, not input_dim
                torch.nn.BatchNorm1d(projection_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
        else:
            self.projection = None
        
        # Main encoder (match training: LayerNorm, not BatchNorm)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # Projection is not used in classification forward pass
        h = self.encoder(x)
        return self.classifier(h)


@app.on_event("startup")
async def load_model():
    """Load the trained neural model on startup"""
    global model, classifier_model, scaler_mean, scaler_std, instruction, hyperparams, metadata, use_stylometric
    
    print("Loading model...")
    
    # Device detection: CUDA (NVIDIA) → MPS (Apple) → CPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple GPU (MPS) detected")
    else:
        device = "cpu"
        print("No GPU detected, using CPU")
    
    print(f"Using device: {device.upper()}")

    # Load trained neural model (.pt file) saved by trustedText.py
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "contrastive_classifier.pt"
    
    if not model_path.exists():
        # Fallback to old pickle format for backward compatibility
        old_model_path = project_root / "models" / "knn_classifier.pkl"
        if old_model_path.exists():
            raise RuntimeError(
                f"Found old KNN model at {old_model_path} but this API requires the new neural model. "
                f"Please run 'python trustedText.py' to train and save the new contrastive model."
            )
        raise RuntimeError(
            f"Trained model not found at {model_path}. Please run 'python trustedText.py' to train and save the model first."
        )

    print(f"Loading trained neural classifier from {model_path}...")
    
    # Load checkpoint (weights_only=False since we trust our own checkpoint file)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Restore preprocessing parameters
    scaler_mean = checkpoint["scaler_mean"]
    scaler_std = checkpoint["scaler_std"]
    hyperparams = checkpoint["hyperparams"]
    metadata = {
        "human_count": checkpoint.get("human_count", 0),
        "ai_count": checkpoint.get("ai_count", 0),
    }
    
    instruction = checkpoint.get(
        "instruction",
        "Analyze the writing style, vocabulary diversity, and sentence structure patterns. Identify if this text was written by a human or generated by AI.\nText: "
    )
    model_id = checkpoint.get("model_id", "Qwen/Qwen3-Embedding-0.6B")
    max_seq_len = int(checkpoint.get("max_seq_length", os.getenv("MAX_SEQ_LEN", "320")))
    use_stylometric = checkpoint.get("config", {}).get("use_stylometric", True)

    # Check if model has projection layer and infer its dimensions from state dict
    state_dict = checkpoint["model_state"]
    has_projection = any(key.startswith("projection.") for key in state_dict.keys())
    
    projection_dim = None
    if has_projection:
        # Infer projection output dimension from the saved weights
        # projection.0.weight has shape [out_features, in_features]
        projection_dim = state_dict["projection.0.weight"].shape[0]
        print(f"Detected projection layer: {hyperparams['input_dim']}D -> {projection_dim}D")
    
    # Reconstruct and load neural classifier
    classifier_model = ContrastiveClassifier(
        input_dim=hyperparams["input_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
        projection_dim=projection_dim
    ).to(device)
    
    # Load state dict
    classifier_model.load_state_dict(state_dict)
    classifier_model.eval()
    
    arch_info = f"-> {projection_dim}D projection -> {hyperparams['hidden_dim']}D hidden" if has_projection else f"-> {hyperparams['hidden_dim']}D hidden"
    print(f"Neural classifier loaded: {hyperparams['input_dim']}D input {arch_info}")

    # Load embedding model
    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,
    )
    model.max_seq_length = max_seq_len

    print("Model loaded successfully")
    print(f"Training samples: {metadata.get('human_count', 0)} human, {metadata.get('ai_count', 0)} AI")
    print(f"Features: {'Semantic + Stylometric' if use_stylometric else 'Semantic only'}")

def encode_text(text: str):
    """Encode text with semantic and optional stylometric features"""
    # Semantic embedding
    prompt = instruction + text
    semantic_emb = model.encode([prompt], normalize_embeddings=True, convert_to_tensor=False)
    semantic_emb = np.array(semantic_emb, dtype=np.float32)
    
    if not use_stylometric:
        return semantic_emb[0]
    
    # Stylometric features
    style_features = StylometricFeatures.extract([text])
    style_mean = np.mean(style_features, axis=0)
    style_std = np.std(style_features, axis=0) + 1e-8
    style_features = (style_features - style_mean) / style_std
    
    # Concatenate
    combined = np.hstack([semantic_emb, style_features])
    return combined[0]

def predict_text(text: str):
    """Make prediction on text"""
    if classifier_model is None or model is None:
        raise RuntimeError("Model not loaded")
    
    # Encode
    features = encode_text(text)
    features_norm = (features - scaler_mean) / scaler_std
    
    # Predict
    device = next(classifier_model.parameters()).device
    x = torch.FloatTensor(features_norm).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classifier_model(x)
        probs = torch.softmax(logits, dim=1)
        ai_prob = float(probs[0][1])
        prediction = 1 if ai_prob >= 0.5 else 0
        
    return {
        "label": "AI" if prediction == 1 else "Human",
        "ai_probability": ai_prob,
        "human_probability": 1 - ai_prob,
        "confidence": "High" if abs(ai_prob - 0.5) > 0.4 else "Medium" if abs(ai_prob - 0.5) > 0.2 else "Low"
    }

@app.get("/")
async def root():
    """Serve the HTML interface"""
    html_path = Path(__file__).resolve().parent / "index.html"
    if not html_path.exists():
        return {"message": "TrustedText API is running", "version": "1.0.0"}
    return FileResponse(html_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and classifier_model is not None,
        "training_samples": {
            "human": metadata.get("human_count", 0),
            "ai": metadata.get("ai_count", 0),
            "total": metadata.get("human_count", 0) + metadata.get("ai_count", 0)
        }
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_text(request: TextRequest):
    """Detect if text is AI-generated"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")
    
    start_time = time.time()
    
    try:
        result = predict_text(request.text)
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            label=result["label"],
            ai_probability=round(result["ai_probability"], 4),
            confidence=result["confidence"],
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
