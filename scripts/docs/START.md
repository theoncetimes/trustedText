# TrustedText Web Interface Usage Guide

This guide will help you run the web interface for the AI text detection model.

## Prerequisites

1. You must have already trained the model by running:
   ```bash
   python -m trustedtext.train
   ```
   This creates the necessary embedding cache in `data/.cache/`

2. Install the required dependencies (if not already installed):
   ```bash
   uv sync
   ```

## Running the Web Interface

### Step 1: Start the API Server

Run the FastAPI backend server:

```bash
python app/api.py
```

Or using uvicorn directly:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Or from the project root:

```bash
cd app
python api.py
```

You should see output like:
```
Loading model...
Using device: MPS
Loading cached embeddings from embeddings_xxx.npz...
Model loaded successfully with XX training samples
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Open the Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

You should see the TrustedText interface with:
- A text input area
- K-Neighbors and Threshold controls
- An "Analyze Text" button

### Step 3: Test the Detector

1. Enter or paste text into the text area
2. (Optional) Adjust the parameters:
   - **K-Neighbors**: Number of nearest neighbors to consider (default: 3)
   - **Threshold**: Probability threshold for AI classification (default: 0.5)
3. Click "Analyze Text" or press Enter
4. View the results showing:
   - Label (Human or AI)
   - AI probability percentage
   - Confidence level
   - Processing time

## API Endpoints

### `GET /`
Serves the HTML interface

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "training_samples": 42
}
```

### `POST /detect`
Detect if text is AI-generated

**Request:**
```json
{
  "text": "Your text here",
  "k": 3,
  "threshold": 0.5
}
```

**Response:**
```json
{
  "label": "AI",
  "ai_probability": 0.8532,
  "confidence": "High",
  "processing_time_ms": 156.42
}
```

## Parameters Explained

- **K-Neighbors**: Higher values make the model consider more training examples, which can smooth out predictions but may reduce sensitivity to outliers.

- **Threshold**: The probability cutoff for AI classification. Lower values (e.g., 0.3) make the model more sensitive to AI text, while higher values (e.g., 0.7) require stronger evidence.

## Troubleshooting

### "No cached embeddings found"
Run `python -m trustedtext.train` first to train the model and create embeddings.

### "Failed to connect to server"
Make sure the API server is running on port 8000. Check the terminal for error messages.

### Port 8000 already in use
Change the port in `app/api.py` or when running uvicorn:
```bash
uvicorn app.api:app --port 8001
```
Then access at `http://localhost:8001`

## Advanced Usage

### Using the API Programmatically

```python
import requests

response = requests.post(
    "http://localhost:8000/detect",
    json={
        "text": "This is a sample text to analyze.",
        "k": 5,
        "threshold": 0.5
    }
)

result = response.json()
print(f"Label: {result['label']}")
print(f"AI Probability: {result['ai_probability']}")
```

### Using curl

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "k": 3, "threshold": 0.5}'
```
