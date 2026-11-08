"""
Heavy shared resources (embedding model, causal LMs) with per-document disk
caches, so repeated folds and seeds never recompute the same document.
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from trustedtext.pipeline import Config, DeviceManager, Embedding

logger = logging.getLogger(__name__)

LM_FEATURES = ("log_likelihood", "log_rank", "entropy", "fast_detectgpt")


def _sha(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def device() -> str:
    return DeviceManager.get_device()


class DocCache:
    """sha(text) -> np.ndarray, persisted as a single .npz file."""

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, np.ndarray] = dict(np.load(path)) if path.exists() else {}
        self._dirty = False

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._data.get(key)

    def put(self, key: str, value: np.ndarray) -> None:
        self._data[key] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **self._data)
        tmp.replace(self.path)
        self._dirty = False


class ChunkEncoder:
    """Embeds each document as a (n_chunks, dim) matrix, as in pipeline.Embedding."""

    def __init__(
        self, model_id: str, instruction: str, max_seq_len: int, batch_size: int, cache_dir: Path
    ):
        self.config = Config(cache_dir=cache_dir)
        self.config.model_id = model_id
        self.config.instruction = instruction
        self.config.max_seq_len = max_seq_len
        self.config.embed_batch_size = batch_size
        self._embedding = Embedding(self.config, device())
        # Device is part of the key: CUDA encodes in fp16, other devices in fp32
        key = _sha(model_id, instruction, max_seq_len, device(), "chunking-v1")
        self._cache = DocCache(cache_dir / f"chunks_{key[:16]}.npz")

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        keys = [_sha(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if self._cache.get(k) is None]
        if missing:
            if self._embedding.model is None:
                self._embedding.load()
            logger.info(f"Embedding {len(missing)} new documents with {self.config.model_id}")
            chunks, doc_ids = self._embedding.chunk_texts([texts[i] for i in missing])
            embeddings = self._embedding.encode(chunks)
            for j, i in enumerate(missing):
                self._cache.put(keys[i], embeddings[doc_ids == j])
            self._cache.save()
        return [self._cache.get(k) for k in keys]


@lru_cache(maxsize=None)
def causal_lm(model_id: str) -> Tuple[torch.nn.Module, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading causal LM {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device() == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device()).eval()
    return model, tokenizer


def _next_token_logits(model_id: str, text: str, max_tokens: int):
    model, tokenizer = causal_lm(model_id)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = ids.input_ids.to(model.device)
    logits = model(ids).logits[0, :-1].float()
    return logits, ids[0, 1:]


class LMFeatureScorer:
    """Per-document zero-shot statistics from one causal LM (see LM_FEATURES)."""

    def __init__(self, model_id: str, max_tokens: int, cache_dir: Path):
        self.model_id = model_id
        self.max_tokens = max_tokens
        key = _sha(model_id, max_tokens, "lm-features-v1")
        self._cache = DocCache(cache_dir / f"lm_{key[:16]}.npz")

    def features(self, texts: List[str]) -> np.ndarray:
        rows = []
        for text in texts:
            key = _sha(text)
            if self._cache.get(key) is None:
                self._cache.put(key, self._compute(text))
            rows.append(self._cache.get(key))
        self._cache.save()
        return np.vstack(rows)

    @torch.no_grad()
    def _compute(self, text: str) -> np.ndarray:
        logits, target = _next_token_logits(self.model_id, text, self.max_tokens)
        lprobs = torch.log_softmax(logits, dim=-1)
        probs = lprobs.exp()
        token_ll = lprobs.gather(-1, target[:, None]).squeeze(-1)
        rank = (lprobs > token_ll[:, None]).sum(-1) + 1

        # Fast-DetectGPT analytic criterion (Bao et al., 2024) with the
        # scoring model doubling as the sampling model
        expected_ll = (probs * lprobs).sum(-1)
        var_ll = (probs * lprobs.square()).sum(-1) - expected_ll.square()
        fast_detectgpt = (token_ll.sum() - expected_ll.sum()) / var_ll.sum().sqrt()

        return np.array(
            [
                token_ll.mean().item(),
                rank.float().log().mean().item(),
                (-expected_ll).mean().item(),
                fast_detectgpt.item(),
            ],
            dtype=np.float32,
        )


class BinocularsScorer:
    """Binoculars score (Hans et al., 2024): log-perplexity / cross-perplexity."""

    def __init__(self, observer_id: str, performer_id: str, max_tokens: int, cache_dir: Path):
        self.observer_id = observer_id
        self.performer_id = performer_id
        self.max_tokens = max_tokens
        key = _sha(observer_id, performer_id, max_tokens, "binoculars-v1")
        self._cache = DocCache(cache_dir / f"binoculars_{key[:16]}.npz")

    def scores(self, texts: List[str]) -> np.ndarray:
        out = []
        for text in texts:
            key = _sha(text)
            if self._cache.get(key) is None:
                self._cache.put(key, np.array([self._compute(text)], dtype=np.float32))
            out.append(self._cache.get(key)[0])
        self._cache.save()
        return np.asarray(out)

    @torch.no_grad()
    def _compute(self, text: str) -> float:
        # Both models must share a tokenizer, so the same ids feed both
        observer_logits, target = _next_token_logits(self.observer_id, text, self.max_tokens)
        performer_logits, _ = _next_token_logits(self.performer_id, text, self.max_tokens)
        performer_lprobs = torch.log_softmax(performer_logits, dim=-1)
        log_ppl = -performer_lprobs.gather(-1, target[:, None]).mean()
        x_log_ppl = -(torch.softmax(observer_logits, dim=-1) * performer_lprobs).sum(-1).mean()
        return (log_ppl / x_log_ppl).item()


@lru_cache(maxsize=None)
def get_encoder(
    model_id: str, instruction: str, max_seq_len: int, batch_size: int, cache_dir: Path
) -> ChunkEncoder:
    return ChunkEncoder(model_id, instruction, max_seq_len, batch_size, cache_dir)


@lru_cache(maxsize=None)
def get_lm_scorer(model_id: str, max_tokens: int, cache_dir: Path) -> LMFeatureScorer:
    return LMFeatureScorer(model_id, max_tokens, cache_dir)


@lru_cache(maxsize=None)
def get_binoculars(
    observer_id: str, performer_id: str, max_tokens: int, cache_dir: Path
) -> BinocularsScorer:
    return BinocularsScorer(observer_id, performer_id, max_tokens, cache_dir)
