import time
import copy
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Optional: UMAP for better visualization
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if not UMAP_AVAILABLE:
    logger.warning("UMAP not installed. Install with: pip install umap-learn")


@dataclass
class Config:
    """Configuration for the AI Text Detector."""

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    max_seq_len: int = 320

    embed_batch_size: int = 4
    random_state: int = 42
    data_dir: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path("data/.cache"))
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # Improved instruction focused on stylistic differences
    instruction: str = """Analyze the writing style, punctuation patterns, lexical diversity, and sentence rhythm. 
Distinguish between human text (natural variability, inconsistent flow) and AI text (uniform structure, predictable patterns).
Text: """

    # Neural classifier params
    hidden_dim: int = 512
    dropout: float = 0.4
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 16
    contrastive_weight: float = (
        0.5  # Weight for contrastive loss vs classification loss
    )
    temperature: float = 0.1  # For contrastive loss

    def __post_init__(self):
        load_dotenv()
        self.max_seq_len = int(os.getenv("MAX_SEQ_LEN", self.max_seq_len))
        self.embed_batch_size = int(
            os.getenv("EMBED_BATCH_SIZE", self.embed_batch_size)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


class DeviceManager:
    """Handles device detection and configuration."""

    @staticmethod
    def get_device() -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple GPU (MPS) detected")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        return device


# ────────────────────────────────────────────────
# EMBEDDING
# ────────────────────────────────────────────────
class Embedding:
    """Manages the sentence transformer model and text encoding."""

    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        self._semantic_dim: Optional[int] = None

    def load(self) -> None:
        """Load the sentence transformer model."""
        logger.info(f"Loading model {self.config.model_id}...")
        start_load = time.time()
        self.model = SentenceTransformer(
            self.config.model_id, device=self.device, trust_remote_code=True
        )
        self.model.max_seq_length = self.config.max_seq_len
        self._semantic_dim = self.model.get_sentence_embedding_dimension()
        self.model.to(torch.float16 if self.device == "cuda" else torch.float32)
        load_time = time.time() - start_load
        logger.info(f"Model loaded in {load_time:.1f}s | Dim: {self._semantic_dim}")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("Model not loaded")
        return self._embedding_dim

    def chunk_texts(
        self, texts: List[str], min_chunk_tokens: int = 30
    ) -> Tuple[List[str], np.ndarray]:
        """
        Split each document into token windows that fit within max_seq_len
        (accounting for the instruction prefix).

        Returns:
            (chunks, doc_ids) where doc_ids[i] is the index in `texts`
            that chunks[i] came from.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokenizer = self.model.tokenizer
        # We tokenize full documents and slice them ourselves, so the
        # tokenizer's max-length warning does not apply here
        tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = True
        instr_len = len(
            tokenizer(self.config.instruction, add_special_tokens=False)["input_ids"]
        )
        # Small margin for special tokens added during encoding
        window = max(32, self.config.max_seq_len - instr_len - 8)

        chunks: List[str] = []
        doc_ids: List[int] = []
        for doc_idx, text in enumerate(texts):
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            pieces = [
                token_ids[start : start + window]
                for start in range(0, len(token_ids), window)
            ]
            # Drop a tiny trailing remainder unless it's the only chunk
            if len(pieces) > 1 and len(pieces[-1]) < min_chunk_tokens:
                pieces = pieces[:-1]
            for piece in pieces:
                chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
                doc_ids.append(doc_idx)

        return chunks, np.asarray(doc_ids, dtype=np.int64)

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode texts with semantic embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Semantic embeddings
        total = len(texts)
        prompt_texts = [self.config.instruction + t for t in texts]

        semantic_embeddings = []
        batch_size = self.config.embed_batch_size

        for start in range(0, total, batch_size):
            batch = prompt_texts[start : start + batch_size]
            batch_emb = self.model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
            )
            semantic_embeddings.append(batch_emb)

            if show_progress and total >= 50:
                logger.info(f"Encoded {min(start + batch_size, total)}/{total}")

        semantic_embeddings = np.vstack(semantic_embeddings).astype(np.float32)

        self._embedding_dim = self._semantic_dim

        return semantic_embeddings


# ────────────────────────────────────────────────
# CONTRASTIVE NEURAL CLASSIFIER
# ────────────────────────────────────────────────
class ContrastiveClassifier(nn.Module):
    """Neural network with contrastive learning capabilities."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, return_embedding=False):
        h = self.encoder(x)
        if return_embedding:
            return F.normalize(self.projection(h), dim=1)
        return self.classifier(h)


class NeuralClassifier:
    """Manages neural classifier with contrastive learning."""

    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        self.model: Optional[ContrastiveClassifier] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.n_splits: int = 0
        self.input_dim = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score normalization."""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def _contrastive_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
    ):
        """
        Supervised contrastive loss.
        Pulls embeddings of same class together, pushes different classes apart.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / temperature

        # Mask out self-similarity (large finite value to avoid inf/nan in exp/log)
        mask = torch.eye(len(labels), device=labels.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Create positive pairs mask (same class, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().masked_fill(mask, 0)

        # Compute log_prob
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )

        # Mean of log probability over positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)

        return -mean_log_prob_pos.mean()

    def _hard_negative_mining(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int = 5
    ) -> np.ndarray:
        """Identify hard examples (AI text that looks human or vice versa)."""
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_model.fit(X)

        distances, indices = nn_model.kneighbors(X)
        hard_indices = []

        for i, (label, neighbors) in enumerate(zip(y, indices)):
            neighbor_labels = y[neighbors]
            # If majority of neighbors are opposite class, it's a hard example
            opposite_ratio = np.sum(neighbor_labels != label) / n_neighbors
            if opposite_ratio >= 0.5:
                hard_indices.append(i)

        if hard_indices:
            logger.info(
                f"Found {len(hard_indices)} hard examples ({len(hard_indices)/len(y):.1%})"
            )

        return np.array(hard_indices)

    def _split_by_document(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        val_fraction: float = 0.15,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Hold out a validation set split at the document level, so chunks of
        the same document never appear on both sides.
        Returns None if the dataset is too small to split.
        """
        docs = np.unique(groups)
        doc_labels = np.array([y[groups == d][0] for d in docs])
        class_counts = np.bincount(doc_labels, minlength=2)

        if len(docs) < 10 or class_counts.min() < 2:
            return None

        train_docs, val_docs = train_test_split(
            docs,
            test_size=val_fraction,
            stratify=doc_labels,
            random_state=self.config.random_state,
        )
        train_mask = np.isin(groups, train_docs)
        return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[ContrastiveClassifier, np.ndarray, np.ndarray, float]:
        """
        Full training procedure (hard-negative weights + contrastive loss),
        with early stopping on validation macro-F1.
        Returns (model, scaler_mean, scaler_std, best_val_f1).
        """
        torch.manual_seed(self.config.random_state)

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        # Identify hard negatives for sampling weight
        hard_indices = self._hard_negative_mining(X_train_norm, y_train)

        # Create sample weights (emphasize hard examples)
        sample_weights = np.ones(len(y_train))
        if len(hard_indices) > 0:
            sample_weights[hard_indices] = 2.0  # Double weight for hard examples

        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        model = ContrastiveClassifier(
            X_train.shape[1], self.config.hidden_dim, self.config.dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

        best_f1 = -1.0
        best_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0

            for batch_x, batch_y, batch_w in loader:
                optimizer.zero_grad()

                # Forward
                logits = model(batch_x)
                emb = model(batch_x, return_embedding=True)

                # Combined loss
                ce_loss = (criterion(logits, batch_y) * batch_w).mean()

                contr_loss = self._contrastive_loss(
                    emb, batch_y, self.config.temperature
                )
                loss = ce_loss + self.config.contrastive_weight * contr_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)

            # Early stopping on validation macro-F1
            model.eval()
            with torch.no_grad():
                val_preds = torch.argmax(model(X_val_tensor), dim=1).cpu().numpy()
            val_f1 = precision_recall_fscore_support(
                y_val, val_preds, average="macro", zero_division=0
            )[2]

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 20 == 0 or patience_counter == 0):
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}, "
                    f"Loss: {avg_loss:.4f}, Val F1: {val_f1:.3f}"
                )

            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, mean, std, best_f1

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        min_class_count: int,
    ) -> None:
        """
        Document-level stratified cross-validation using the same training
        procedure as fit(). Folds are split by document, and fold scores are
        computed on document-level predictions (mean chunk probability).
        """
        docs = np.unique(groups)
        doc_labels = np.array([y[groups == d][0] for d in docs])

        if len(docs) < 20 or min_class_count < 2:
            logger.info("Dataset too small for cross-validation")
            return

        self.n_splits = min(5, min_class_count)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scores = []
        for fold, (train_doc_idx, val_doc_idx) in enumerate(
            skf.split(docs, doc_labels)
        ):
            train_mask = np.isin(groups, docs[train_doc_idx])
            X_fold, y_fold = X[train_mask], y[train_mask]
            groups_fold = groups[train_mask]

            # Inner document-level split for early stopping
            split = self._split_by_document(X_fold, y_fold, groups_fold)
            if split is None:
                split = (X_fold, y_fold, X_fold, y_fold)

            model, mean, std, _ = self._train_model(*split, verbose=False)

            # Score on held-out documents (aggregate chunk probs per doc)
            model.eval()
            val_docs = docs[val_doc_idx]
            X_val_norm = (X[~train_mask] - mean) / std
            val_groups = groups[~train_mask]
            with torch.no_grad():
                probs = F.softmax(
                    model(torch.FloatTensor(X_val_norm).to(self.device)), dim=1
                )[:, 1].cpu().numpy()

            doc_preds = np.array(
                [int(probs[val_groups == d].mean() >= 0.5) for d in val_docs]
            )
            f1 = precision_recall_fscore_support(
                doc_labels[val_doc_idx], doc_preds, average="macro", zero_division=0
            )[2]
            scores.append(f1)
            logger.info(f"CV fold {fold + 1}/{self.n_splits}: doc-level F1 {f1:.3f}")

        self.cv_scores = np.array(scores)
        logger.info(
            f"CV F1-macro (doc-level): {self.cv_scores.mean():.3f} "
            f"(±{self.cv_scores.std():.3f})"
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> None:
        """Train with contrastive learning and validation-based early stopping."""
        logger.info("Preparing data...")
        self.input_dim = X.shape[1]

        split = (
            self._split_by_document(X, y, groups) if groups is not None else None
        )
        if split is None:
            logger.warning(
                "Too few documents for a validation split; "
                "early stopping will use training data"
            )
            split = (X, y, X, y)
        else:
            logger.info(
                f"Holding out {len(split[3])} chunks for validation-based early stopping"
            )

        logger.info(f"Training neural classifier for {self.config.epochs} epochs...")
        self.model, self.scaler_mean, self.scaler_std, best_f1 = self._train_model(
            *split
        )
        logger.info(f"Training complete (best val F1: {best_f1:.3f})")

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and probabilities.
        Returns: (predictions, ai_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        X_norm = self._normalize(embeddings, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            ai_probs = probs[:, 1].cpu().numpy()

        return predictions, ai_probs

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Get projected embeddings from the contrastive head."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        X_norm = self._normalize(embeddings, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            # return_embedding=True returns the normalized projection
            proj = self.model(X_tensor, return_embedding=True)
            return proj.cpu().numpy()

    def save(self, path: Path, metadata: Dict[str, Any]) -> None:
        """Save model and preprocessing params."""
        if self.model is None:
            raise RuntimeError("No model to save")

        save_dict = {
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "hyperparams": {
                "input_dim": self.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
            },
            "config": {
                "instruction": self.config.instruction,
                "model_id": self.config.model_id,
                "max_seq_len": self.config.max_seq_len,
            },
            **metadata,
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> Dict[str, Any]:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)

        # Reconstruct model
        hp = checkpoint["hyperparams"]
        self.model = ContrastiveClassifier(
            hp["input_dim"], hp["hidden_dim"], hp["dropout"]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        self.scaler_mean = checkpoint["scaler_mean"]
        self.scaler_std = checkpoint["scaler_std"]

        logger.info(f"Model loaded from {path}")
        return checkpoint


# ────────────────────────────────────────────────
# STORAGE (Minor updates)
# ────────────────────────────────────────────────
class Storage:
    """Handles data loading, caching, and preprocessing."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _read_txt_file(path: Path) -> List[str]:
        content = path.read_text(encoding="utf-8").strip()
        return [content] if content else []

    def _load_texts_from_folder(self, folder: Path) -> List[str]:
        texts = []
        for txt_file in sorted(folder.glob("*.txt")):
            texts.extend(self._read_txt_file(txt_file))
        return texts

    def load_texts(self) -> Tuple[List[str], List[str]]:
        human_dir = self.config.data_dir / "human"
        ai_dir = self.config.data_dir / "ai"
        human_file = self.config.data_dir / "human.txt"
        ai_file = self.config.data_dir / "ai.txt"

        if human_dir.exists() and ai_dir.exists():
            human_texts = self._load_texts_from_folder(human_dir)
            ai_texts = self._load_texts_from_folder(ai_dir)
        elif human_file.exists() and ai_file.exists():
            human_texts = self._read_txt_file(human_file)
            ai_texts = self._read_txt_file(ai_file)
        else:
            raise FileNotFoundError(
                "Expected data in data/human/*.txt and data/ai/*.txt, "
                "or data/human.txt and data/ai.txt."
            )

        if not human_texts or not ai_texts:
            raise ValueError("Both human and AI text sets must be non-empty.")

        return human_texts, ai_texts

    def _compute_signature(self) -> str:
        """Compute dataset signature for cache invalidation."""
        file_stats = []
        for path in sorted(self.config.data_dir.rglob("*.txt")):
            stat = path.stat()
            file_stats.append(
                {
                    "path": str(path.relative_to(self.config.data_dir)),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        payload = {
            "files": file_stats,
            "instruction": self.config.instruction,
            "model_id": self.config.model_id,
            "max_seq_len": self.config.max_seq_len,
            "chunking": "v1",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest

    def load_or_create_embeddings(
        self, embedding: Embedding
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Load embeddings from cache or create them.
        Documents are split into chunks; doc_ids maps each chunk back to its
        source document. human_count/ai_count are document counts.
        """
        signature = self._compute_signature()
        cache_path = self.config.cache_dir / f"embeddings_{signature}.npz"

        if cache_path.exists():
            logger.info("Loading cached embeddings...")
            cache = np.load(cache_path)
            embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
            labels = np.asarray(cache["labels"], dtype=np.int64)
            doc_ids = np.asarray(cache["doc_ids"], dtype=np.int64)
            human_count = int(cache["human_count"])
            ai_count = int(cache["ai_count"])
        else:
            logger.info("Loading texts from data folder...")
            human_texts, ai_texts = self.load_texts()
            all_texts = human_texts + ai_texts
            doc_labels = np.asarray(
                [0] * len(human_texts) + [1] * len(ai_texts), dtype=np.int64
            )
            human_count = len(human_texts)
            ai_count = len(ai_texts)

            chunks, doc_ids = embedding.chunk_texts(all_texts)
            labels = doc_labels[doc_ids]

            logger.info(
                f"Dataset stats: Human={human_count}, AI={ai_count} documents "
                f"-> {len(chunks)} chunks"
            )
            logger.info("Encoding dataset...")

            start_encode = time.time()
            embeddings = embedding.encode(chunks, show_progress=True)
            encode_time = time.time() - start_encode
            logger.info(f"Encoded {len(chunks)} chunks in {encode_time:.1f}s")

            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                labels=labels,
                doc_ids=doc_ids,
                human_count=human_count,
                ai_count=ai_count,
            )

            del human_texts, ai_texts, all_texts, chunks
            gc.collect()

        return embeddings, labels, doc_ids, human_count, ai_count


# ────────────────────────────────────────────────
# EVALUATION
# ────────────────────────────────────────────────
class Evaluation:
    """Handles test set evaluation."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.test_dir = data_dir / "test"

    def load_test_data(self) -> Optional[Tuple[List[str], np.ndarray, int, int]]:
        test_human_dir = self.test_dir / "human"
        test_ai_dir = self.test_dir / "ai"

        if not (test_human_dir.exists() and test_ai_dir.exists()):
            return None

        storage = Storage(Config(data_dir=self.data_dir))
        human_texts = storage._load_texts_from_folder(test_human_dir)
        ai_texts = storage._load_texts_from_folder(test_ai_dir)
        texts = human_texts + ai_texts

        if not texts:
            return None

        labels = np.asarray(
            [0] * len(human_texts) + [1] * len(ai_texts), dtype=np.int64
        )
        return texts, labels, len(human_texts), len(ai_texts)

    def evaluate(self, classifier, embedding, texts, true_labels):
        """Comprehensive evaluation. Predicts per chunk, aggregates per document."""
        chunks, doc_ids = embedding.chunk_texts(texts)
        logger.info(f"Encoding {len(texts)} test samples ({len(chunks)} chunks)...")
        test_embeddings = embedding.encode(chunks, show_progress=False)

        _, chunk_probs = classifier.predict(test_embeddings)

        # Mean chunk probability per document
        ai_probs = np.array(
            [
                chunk_probs[doc_ids == i].mean() if np.any(doc_ids == i) else 0.5
                for i in range(len(texts))
            ]
        )
        predictions = (ai_probs >= 0.5).astype(int)

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

        # Find misclassified examples
        misclassified = []
        for i, (pred, true, prob) in enumerate(zip(predictions, true_labels, ai_probs)):
            if pred != true:
                misclassified.append(
                    {
                        "index": i,
                        "text": texts[i][:100],
                        "true": "AI" if true == 1 else "Human",
                        "pred": "AI" if pred == 1 else "Human",
                        "confidence": float(prob if pred == 1 else 1 - prob),
                    }
                )

        if misclassified:
            logger.info(f"\nMisclassified examples ({len(misclassified)}):")
            for ex in misclassified[:3]:  # Show first 3
                logger.info(
                    f"  {ex['text'][:60]}... | True: {ex['true']} | Pred: {ex['pred']} (conf: {ex['confidence']:.2f})"
                )

        return {
            "accuracy": float(accuracy),
            "precision_human": float(precision[0]),
            "precision_ai": float(precision[1]),
            "recall_human": float(recall[0]),
            "recall_ai": float(recall[1]),
            "f1_human": float(f1[0]),
            "f1_ai": float(f1[1]),
            "confusion_matrix": cm.tolist(),
            "misclassified": misclassified,
        }

    def print_results(self, metrics: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(
            f"\nClass 0 (Human) - P: {metrics['precision_human']:.3f}, R: {metrics['recall_human']:.3f}, F1: {metrics['f1_human']:.3f}"
        )
        print(
            f"Class 1 (AI)    - P: {metrics['precision_ai']:.3f}, R: {metrics['recall_ai']:.3f}, F1: {metrics['f1_ai']:.3f}"
        )

        cm = metrics["confusion_matrix"]
        print("\nConfusion Matrix:")
        print("                Pred Human   Pred AI")
        print(f"True Human      {cm[0][0]:6d}      {cm[0][1]:6d}")
        print(f"True AI         {cm[1][0]:6d}      {cm[1][1]:6d}")
        print(f"{'='*60}")


# ────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ────────────────────────────────────────────────
class TrustedText:
    """Main orchestrator with enhanced detection capabilities."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = DeviceManager.get_device()
        self.embedding = Embedding(self.config, self.device)
        self.storage = Storage(self.config)
        self.classifier = NeuralClassifier(self.config, self.device)
        self.evaluation = Evaluation(self.config.data_dir)

        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.doc_ids: Optional[np.ndarray] = None
        self.human_count: int = 0
        self.ai_count: int = 0

    def setup(self):
        """Initialize and load models."""
        self.embedding.load()

    def train(self):
        """Execute full training pipeline."""
        logger.info("Preparing dataset...")

        self.embeddings, self.labels, self.doc_ids, self.human_count, self.ai_count = (
            self.storage.load_or_create_embeddings(self.embedding)
        )

        logger.info(
            f"Dataset ready: Human={self.human_count}, AI={self.ai_count} documents "
            f"({len(self.embeddings)} chunks)"
        )

        min_class_count = min(self.human_count, self.ai_count)
        self.classifier.validate(
            self.embeddings, self.labels, self.doc_ids, min_class_count
        )

        logger.info(f"\nTraining on {len(self.embeddings)} chunks...")
        self.classifier.fit(self.embeddings, self.labels, self.doc_ids)

    def visualize(
        self, method: str = "umap", max_points: int = 500, use_projection: bool = False
    ) -> None:
        """
        Visualize embeddings with UMAP or t-SNE.

        Args:
            method: 'umap' (preferred) or 'tsne'
            max_points: Max points to plot
            use_projection: If True, use trained model's projected embeddings
        """
        if self.embeddings is None or self.labels is None:
            raise RuntimeError("No embeddings found. Run train() first.")

        X = self.embeddings
        y = self.labels

        if len(X) > max_points:
            rng = np.random.RandomState(self.config.random_state)
            idx = rng.choice(len(X), size=max_points, replace=False)
            X = X[idx]
            y = y[idx]

        title_suffix = " (Raw)"
        if use_projection:
            logger.info("Projecting embeddings through trained model...")
            X = self.classifier.project(X)
            title_suffix = " (Trained Projection)"

        logger.info(f"Computing {method} projection...")

        actual_method = "t-SNE"
        if method == "umap" and UMAP_AVAILABLE:
            actual_method = "UMAP"
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=self.config.random_state,
                metric="cosine",
            )
        else:
            if method == "umap":
                logger.warning("UMAP not available, falling back to t-SNE")

            # Fix: Removed n_iter parameter (not valid in newer sklearn versions)
            reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(X) - 1),
                random_state=self.config.random_state,
                init="pca",  # Better initialization than random
                learning_rate="auto",  # Auto learning rate for better convergence
            )

        emb_2d = reducer.fit_transform(X)

        plt.figure(figsize=(12, 8))

        scatter = plt.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=y,
            cmap="coolwarm",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        plt.colorbar(scatter, label="Class (0=Human, 1=AI)")
        plt.title(
            f"{actual_method} of Embeddings{title_suffix} (Blue=Human, Red=AI)"
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.show()

        # Also show class separation stats
        human_points = emb_2d[y == 0]
        ai_points = emb_2d[y == 1]

        if len(human_points) > 0 and len(ai_points) > 0:
            centroid_dist = np.linalg.norm(
                np.mean(human_points, axis=0) - np.mean(ai_points, axis=0)
            )
            logger.info(f"Centroid distance between classes: {centroid_dist:.3f}")

    def evaluate(self) -> Optional[Dict[str, Any]]:
        """Evaluate on test set if available."""
        test_data = self.evaluation.load_test_data()

        if test_data is None:
            logger.info(f"\nNo test data found at {self.evaluation.test_dir}")
            return None

        texts, true_labels, test_humans, test_ai = test_data
        logger.info(f"\nEvaluating on {len(texts)} test samples...")

        metrics = self.evaluation.evaluate(
            self.classifier, self.embedding, texts, true_labels
        )

        self.evaluation.print_results(metrics)
        return metrics

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.config.model_dir / "contrastive_classifier.pt"

        metadata = {
            "human_count": self.human_count,
            "ai_count": self.ai_count,
        }
        self.classifier.save(path, metadata)
        return path

    def save_stats(self, test_metrics: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save training statistics to cache for visualize_stats.py.
        Call after train() and optionally after evaluate().
        """
        train_size = self.human_count + self.ai_count
        test_size = 0
        if test_metrics:
            test_size = sum(
                test_metrics["confusion_matrix"][i][j]
                for i in range(2)
                for j in range(2)
            )
        total_samples = train_size + test_size

        stats = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "human_count": self.human_count,
                "ai_count": self.ai_count,
                "train_size": train_size,
                "test_size": test_size,
                "total_samples": total_samples,
            },
            "cross_validation": {
                "scores": (
                    self.classifier.cv_scores.tolist()
                    if self.classifier.cv_scores is not None
                    else []
                ),
                "mean_f1": (
                    float(self.classifier.cv_scores.mean())
                    if self.classifier.cv_scores is not None
                    else 0.0
                ),
                "std_f1": (
                    float(self.classifier.cv_scores.std())
                    if self.classifier.cv_scores is not None
                    else 0.0
                ),
            },
        }
        if test_metrics:
            stats["confusion_matrix"] = test_metrics["confusion_matrix"]
            stats["test_metrics"] = {
                "accuracy": test_metrics["accuracy"],
                "precision_human": test_metrics["precision_human"],
                "precision_ai": test_metrics["precision_ai"],
                "recall_human": test_metrics["recall_human"],
                "recall_ai": test_metrics["recall_ai"],
                "f1_human": test_metrics["f1_human"],
                "f1_ai": test_metrics["f1_ai"],
            }

        stats_path = self.config.cache_dir / "train_stats.json"
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Training stats saved to {stats_path}")
        return stats_path

    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self.config.model_dir / "contrastive_classifier.pt"

        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        self.classifier.load(path)

    def predict(
        self, text: str, return_prob: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """Detect if text is AI-generated (mean probability over chunks)."""
        if self.classifier.model is None:
            raise RuntimeError("No trained model available.")

        if self.embedding.model is None:
            self.embedding.load()

        chunks, _ = self.embedding.chunk_texts([text])
        if not chunks:
            raise ValueError("Text is empty")

        emb = self.embedding.encode(chunks)
        _, probs = self.classifier.predict(emb)
        ai_prob = float(np.mean(probs))

        label = "AI" if ai_prob >= 0.5 else "Human"

        if return_prob:
            return label, ai_prob
        return label

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Detailed analysis of a single text."""
        label, prob = self.predict(text, return_prob=True)

        return {
            "prediction": label,
            "ai_probability": prob,
            "human_probability": 1 - prob,
            "confidence": (
                "High"
                if abs(prob - 0.5) > 0.4
                else "Medium" if abs(prob - 0.5) > 0.2 else "Low"
            ),
        }


if __name__ == "__main__":
    # Configuration
    config = Config(epochs=100, hidden_dim=512)

    # Initialize and train
    detector = TrustedText(config)
    detector.setup()
    detector.train()

    # Visualize raw embeddings (how the base model sees it)
    print("\nVisualizing raw embeddings...")
    detector.visualize(method="umap", use_projection=False)

    # Visualize projected embeddings (how your trained model sees it)
    print("\nVisualizing trained projection...")
    detector.visualize(method="umap", use_projection=True)

    # Evaluate if test data exists
    metrics = detector.evaluate()

    # Save model
    detector.save()

    # Save stats to cache for visualize_stats.py
    detector.save_stats(test_metrics=metrics)

    # Example inference
    sample_text = "ok you are clever honestly. But you forgot who am I. I am Big Kimi(evil smile) so my O is too big that occupy your position. I am sorry about that."
    result = detector.predict(sample_text)
    print(f"\nPrediction: {result}")

    # Detailed analysis
    analysis = detector.analyze_text(sample_text)
    print(f"\nAnalysis: {json.dumps(analysis, indent=2)}")
