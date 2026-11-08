# Uploading to Hugging Face Hub

This guide explains how to upload your trained TrustedText model to Hugging Face Hub for easy sharing and deployment.

## Prerequisites

1. **Train your model** (if not already done):
   ```bash
   python -m trustedtext.train
   ```

2. **Install Hugging Face Hub library**:
   ```bash
   pip install huggingface_hub
   ```
   Or if using uv:
   ```bash
   uv pip install huggingface_hub
   ```

3. **Create a Hugging Face account** (if you don't have one):
   - Visit [https://huggingface.co/join](https://huggingface.co/join)
   - Create a free account

4. **Login to Hugging Face CLI**:
   ```bash
   huggingface-cli login
   ```
   - This will prompt you for your access token
   - Get your token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Choose "Write" access when creating the token

## Quick Upload

### Step 1: Configure the Upload Script

Edit `scripts/upload_to_huggingface.py` and change these settings:

```python
# Configuration
REPO_NAME = "trustedtext-ai-detector"  # Your desired repo name
USERNAME = None  # Will use your logged-in username
PRIVATE = False  # Set to True for private repo
```

### Step 2: Run the Upload Script

```bash
python scripts/upload_to_huggingface.py
```

The script will:
1. ✅ Create a repository on Hugging Face
2. ✅ Generate a comprehensive model card (README.md)
3. ✅ Upload your trained model (`knn_classifier.pkl`)
4. ✅ Upload requirements and usage examples
5. ✅ Provide you with the model URL

### Step 3: View Your Model

After successful upload, visit:
```
https://huggingface.co/theoncetimes/trustedtext-ai-detector
```

## What Gets Uploaded

1. **knn_classifier.pkl** - Your trained model containing:
   - KNN classifier
   - Training embeddings
   - Model metadata (model ID, sequence length, etc.)
   - Training statistics

2. **README.md** - Comprehensive model card with:
   - Model description
   - Installation instructions
   - Usage examples
   - Performance metrics
   - Limitations

3. **requirements.txt** - Dependencies needed to use the model

4. **inference_example.py** - Standalone inference script

## Using Your Uploaded Model

### Download and Use

```python
from huggingface_hub import hf_hub_download
import pickle
from sentence_transformers import SentenceTransformer
import torch

# Download model
model_path = hf_hub_download(
    repo_id="theoncetimes/trustedtext-ai-detector",
    filename="knn_classifier.pkl"
)

# Load and use
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

knn = model_data["knn"]
instruction = model_data["instruction"]

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    model_data["model_id"],
    device=device,
    trust_remote_code=True
)

# Detect function
def detect(text: str):
    prompt = instruction + text
    embedding = embedding_model.encode([prompt], normalize_embeddings=True)
    proba = knn.predict_proba(embedding)[0]
    return "AI" if proba[1] >= 0.5 else "Human", float(proba[1])

# Example
label, prob = detect("Sample text to analyze")
print(f"Label: {label}, AI Probability: {prob:.3f}")
```

## Advanced Options

### Private Repository

To upload as a private model:

```python
PRIVATE = True  # in upload_to_huggingface.py
```

### Custom Repository Name

```python
REPO_NAME = "my-custom-ai-detector"
```

### Update Existing Model

The script automatically updates existing repositories. Just run it again:

```bash
python scripts/upload_to_huggingface.py
```

## Model Card Customization

After uploading, you can edit the model card directly on Hugging Face:

1. Go to your model page
2. Click "Edit model card"
3. Update the README.md with:
   - Your GitHub repository link
   - Additional performance metrics
   - Custom examples
   - Training details

## Sharing Your Model

### Public Model (Default)

Anyone can:
- View your model page
- Download the model
- Use it in their projects

### Private Model

Only you and collaborators can access it.

### Adding Collaborators

1. Go to your model page
2. Settings → Collaborators
3. Add users by username

## Model Versioning

Hugging Face automatically versions your uploads. To update:

1. Retrain your model with new data
2. Run the upload script again
3. Previous versions are preserved

## Troubleshooting

### "Error: Not logged in"

```bash
huggingface-cli login
```

### "Error: Model not found"

Make sure you've trained the model first:
```bash
python -m trustedtext.train
```

### "Error: Permission denied"

Check your access token has "Write" permissions:
- Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Create new token with "Write" access

### "Error: Repository already exists"

The script handles this automatically, but if you want to start fresh:
```bash
huggingface-cli delete theoncetimes/trustedtext-ai-detector
python upload_to_huggingface.py
```

## Best Practices

1. **Train on quality data** - Your model is only as good as your training data
2. **Document performance** - Add metrics to your model card
3. **Include limitations** - Be transparent about where the model works well/poorly
4. **Update regularly** - Retrain and upload as you get more data
5. **Use semantic versioning** - Tag releases (v1.0.0, v1.1.0, etc.)

## Integration with Spaces

You can also deploy your model as a Hugging Face Space:

1. Create a new Space on Hugging Face
2. Choose Gradio or Streamlit
3. Use the inference code from the model card
4. Deploy with a web interface

Example Gradio app:

```python
import gradio as gr
from huggingface_hub import hf_hub_download
import pickle
from sentence_transformers import SentenceTransformer
import torch

# Load model
model_path = hf_hub_download(
    repo_id="theoncetimes/trustedtext-ai-detector",
    filename="knn_classifier.pkl"
)

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

# Initialize
knn = model_data["knn"]
instruction = model_data["instruction"]
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(
    model_data["model_id"],
    device=device,
    trust_remote_code=True
)

def detect(text, k, threshold):
    prompt = instruction + text
    embedding = embedding_model.encode([prompt], normalize_embeddings=True)
    knn.n_neighbors = min(k, len(knn._fit_X))
    proba = knn.predict_proba(embedding)[0]
    label = "AI" if proba[1] >= threshold else "Human"
    return {
        "Label": label,
        "AI Probability": f"{proba[1]:.1%}",
        "Human Probability": f"{proba[0]:.1%}"
    }

# Create interface
demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Textbox(label="Text to Analyze", lines=5),
        gr.Slider(1, 10, value=3, step=1, label="K Neighbors"),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Threshold")
    ],
    outputs=gr.JSON(label="Results"),
    title="TrustedText AI Detector",
    description="Detect if text is AI-generated or human-written"
)

demo.launch()
```

## Resources

- Hugging Face Hub Documentation: [https://huggingface.co/docs/hub](https://huggingface.co/docs/hub)
- Model Card Guide: [https://huggingface.co/docs/hub/model-cards](https://huggingface.co/docs/hub/model-cards)
- Spaces Documentation: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)

## Support

For issues with:
- **TrustedText model**: Open an issue on GitHub
- **Hugging Face upload**: Check [Hugging Face forums](https://discuss.huggingface.co/)
