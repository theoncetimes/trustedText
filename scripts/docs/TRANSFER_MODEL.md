# Model Transfer Guide

This guide explains how to train a model on a GPU machine and transfer it back to another machine.

## Overview

The training process creates a `models/` directory containing the trained KNN classifier and training data. You can transfer this directory between machines to use the trained model without retraining.

## Files Involved

- `models/knn_classifier.pkl` - The trained model (created after training)
- `trustedText.py` - Main training script (saves model automatically)
- `scripts/load_model.py` - Loading script (loads and uses saved model)

## Step-by-Step Process

### On GPU Machine (Training)

1. **Setup environment:**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # Install other dependencies
   pip install sentence-transformers scikit-learn numpy
   ```

2. **Copy your data to the GPU machine:**
   ```bash
   # On your local machine
   scp -r data/ user@gpu-machine:/path/to/trustedText/
   
   # Or use rsync for better performance
   rsync -avz data/ user@gpu-machine:/path/to/trustedText/data/
   ```

3. **Copy the training script:**
   ```bash
   scp trustedText.py user@gpu-machine:/path/to/trustedText/
   ```

4. **Run training on GPU machine:**
   ```bash
   ssh user@gpu-machine
   cd /path/to/trustedText
   python -m trustedtext.train
   ```
   
   You should see:
   ```
   NVIDIA GPU detected: [GPU name]
   Using device: CUDA
   ...
   Model saved to models/knn_classifier.pkl
   ```

5. **Download the trained model:**
   ```bash
   # On your local machine
   scp -r user@gpu-machine:/path/to/trustedText/models/ ./
   
   # Or use rsync
   rsync -avz user@gpu-machine:/path/to/trustedText/models/ ./models/
   ```

### On Local Machine (Inference)

1. **Load and use the model:**
   ```bash
   python scripts/load_model.py
   ```

2. **Use in your code:**

   **Option 1: Using the module**
   ```python
   from trustedText import TrustedText
   
   detector = TrustedText()
   detector.load()
   label, probability = detector.predict("Your text here")
   print(f"Prediction: {label} (AI prob: {probability:.3f})")
   ```

   **Option 2: Using the script**
   ```python
   from scripts.load_model import detect
   
   label, probability = detect("Your text here")
   print(f"Prediction: {label} (AI prob: {probability:.3f})")
   ```

## Alternative: Cloud Storage Transfer

Instead of direct SCP, you can use cloud storage:

### Using S3/MinIO:
```bash
# On GPU machine (after training)
aws s3 cp models/knn_classifier.pkl s3://your-bucket/models/

# On local machine
aws s3 cp s3://your-bucket/models/knn_classifier.pkl models/
```

### Using Google Cloud Storage:
```bash
# On GPU machine
gsutil cp models/knn_classifier.pkl gs://your-bucket/models/

# On local machine
gsutil cp gs://your-bucket/models/knn_classifier.pkl models/
```

### Using Dropbox/Google Drive:
1. On GPU machine: Copy `models/` to synced folder
2. Wait for sync
3. On local machine: Copy from synced folder to project

## What Gets Transferred

The `knn_classifier.pkl` file contains:
- Trained KNN model
- Training embeddings (X_train, y_train)
- Instruction prompt
- Model configuration (model_id, max_seq_length)
- Dataset statistics (human_count, ai_count)

**Size:** Typically 50-500 MB depending on training data size.

## Troubleshooting

### Model file not found
```
FileNotFoundError: Model file not found at models/knn_classifier.pkl
```
**Solution:** Run `python -m trustedtext.train` first or verify the transfer completed successfully.

### Version mismatch
If you get pickle errors, ensure both machines have compatible package versions:
```bash
pip list | grep -E "scikit-learn|numpy|torch"
```

### Different PyTorch versions
The model file is compatible across PyTorch versions (CPU, CUDA, MPS) because it only contains numpy arrays and sklearn models. The sentence transformer model is downloaded separately on each machine.

## Model Size Optimization

If the model file is too large, you can reduce it by:

1. **Training with less data** (if acceptable)
2. **Using compression:**
   ```python
   # In train.py, replace pickle with:
   import joblib
   joblib.dump(model_data, model_path, compress=3)
   
   # In load_model.py, replace pickle with:
   model_data = joblib.load(model_path)
   ```

## Security Notes

- The model file contains your training data embeddings
- Don't share publicly if training data is sensitive
- Transfer over secure channels (SSH, HTTPS)
- Consider encrypting before cloud storage upload

## Quick Reference

```bash
# Train on GPU machine
python -m trustedtext.train  # Creates models/knn_classifier.pkl

# Transfer model
scp -r models/ user@local-machine:/path/to/project/

# Use on local machine
python scripts/load_model.py  # Loads and tests model
```
