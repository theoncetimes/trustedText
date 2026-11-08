import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from trustedtext import TrustedText

project_root = Path(__file__).resolve().parent.parent

app = FastAPI(title="TrustedText API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: TrustedText | None = None


class TextRequest(BaseModel):
    text: str


class DetectionResponse(BaseModel):
    label: str
    ai_probability: float
    confidence: str
    processing_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load the trained classifier and its embedding model on startup."""
    global detector

    model_path = project_root / "models" / "contrastive_classifier.pt"
    if not model_path.exists():
        raise RuntimeError(
            f"Trained model not found at {model_path}. "
            "Run 'python -m trustedtext.train' to train and save the model first."
        )

    detector = TrustedText()
    detector.load(model_path)
    detector.setup()
    print(
        f"Model loaded. Training samples: "
        f"{detector.human_count} human, {detector.ai_count} AI"
    )


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
    loaded = detector is not None
    human = detector.human_count if loaded else 0
    ai = detector.ai_count if loaded else 0
    return {
        "status": "healthy",
        "model_loaded": loaded,
        "training_samples": {"human": human, "ai": ai, "total": human + ai},
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_text(request: TextRequest):
    """Detect if text is AI-generated"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400, detail="Text too long (max 10,000 characters)"
        )

    start_time = time.time()

    try:
        result = detector.analyze_text(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    return DetectionResponse(
        label=result["prediction"],
        ai_probability=round(result["ai_probability"], 4),
        confidence=result["confidence"],
        processing_time_ms=round((time.time() - start_time) * 1000, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
