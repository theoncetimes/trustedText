"""
Detectors share one interface: fit on training documents, then return
(score, prob) for test documents. `score` is higher for more AI-like text;
`prob` is P(AI). Supervised detectors return their probability as both.
Zero-shot detectors only use training labels to calibrate `prob`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustedtext.bench.resources import (
    LM_FEATURES,
    device,
    get_binoculars,
    get_encoder,
    get_lm_scorer,
)
from trustedtext.pipeline import Config, NeuralClassifier

DEFAULT_LM = "Qwen/Qwen2.5-0.5B"
DEFAULT_LM_INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"


class Detector(ABC):
    def fit(self, texts: List[str], labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (score, prob)."""


def _logistic_regression(C: Optional[float], seed: int):
    """Fixed C if given; otherwise C is tuned by inner CV on the training fold only."""
    if C is not None:
        return LogisticRegression(C=C, max_iter=5000)
    return LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 7),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        scoring="roc_auc",
        max_iter=5000,
        l1_ratios=(0.0,),
        use_legacy_attributes=False,
    )


class TfidfDetector(Detector):
    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: Sequence[int] = (1, 2),
        min_df: int = 1,
        C: Optional[float] = None,
        seed: int = 0,
        cache_dir: Optional[Path] = None,
    ):
        self.model = make_pipeline(
            TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=tuple(ngram_range),
                min_df=min_df,
                sublinear_tf=True,
            ),
            _logistic_regression(C, seed),
        )

    def fit(self, texts, labels):
        self.model.fit(texts, labels)

    def predict(self, texts):
        prob = self.model.predict_proba(texts)[:, 1]
        return prob, prob


class _EncoderMixin:
    def _init_encoder(self, model_id, instruction, max_seq_len, batch_size, cache_dir):
        self.encoder = get_encoder(model_id, instruction, max_seq_len, batch_size, Path(cache_dir))


class EmbeddingLRDetector(Detector, _EncoderMixin):
    """Logistic regression on the mean chunk embedding of each document."""

    def __init__(
        self,
        model_id: str = Config.model_id,
        instruction: str = Config.instruction,
        max_seq_len: int = Config.max_seq_len,
        batch_size: int = Config.embed_batch_size,
        C: Optional[float] = None,
        seed: int = 0,
        cache_dir: Path = Path("data/.cache/bench"),
    ):
        self._init_encoder(model_id, instruction, max_seq_len, batch_size, cache_dir)
        self.model = make_pipeline(StandardScaler(), _logistic_regression(C, seed))

    def _features(self, texts):
        return np.vstack([m.mean(axis=0) for m in self.encoder.embed(texts)])

    def fit(self, texts, labels):
        self.model.fit(self._features(texts), labels)

    def predict(self, texts):
        prob = self.model.predict_proba(self._features(texts))[:, 1]
        return prob, prob


class ContrastiveDetector(Detector, _EncoderMixin):
    """The TrustedText classifier: chunk-level MLP with supervised contrastive loss."""

    def __init__(
        self,
        model_id: str = Config.model_id,
        instruction: str = Config.instruction,
        max_seq_len: int = Config.max_seq_len,
        batch_size: int = Config.embed_batch_size,
        hidden_dim: int = Config.hidden_dim,
        dropout: float = Config.dropout,
        learning_rate: float = Config.learning_rate,
        epochs: int = Config.epochs,
        train_batch_size: int = Config.batch_size,
        contrastive_weight: float = Config.contrastive_weight,
        temperature: float = Config.temperature,
        hard_negative_weight: float = Config.hard_negative_weight,
        seed: int = 0,
        cache_dir: Path = Path("data/.cache/bench"),
    ):
        self._init_encoder(model_id, instruction, max_seq_len, batch_size, cache_dir)
        config = Config(cache_dir=Path(cache_dir))
        for key, value in {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": train_batch_size,
            "contrastive_weight": contrastive_weight,
            "temperature": temperature,
            "hard_negative_weight": hard_negative_weight,
            "random_state": seed,
        }.items():
            setattr(config, key, value)
        self.classifier = NeuralClassifier(config, device())

    def _chunks(self, texts):
        mats = self.encoder.embed(texts)
        doc_index = np.repeat(np.arange(len(mats)), [len(m) for m in mats])
        return np.vstack(mats), doc_index

    def fit(self, texts, labels):
        X, doc_index = self._chunks(texts)
        self.classifier.fit(X, np.asarray(labels)[doc_index], doc_index)

    def predict(self, texts):
        X, doc_index = self._chunks(texts)
        _, chunk_prob = self.classifier.predict(X)
        prob = np.array([chunk_prob[doc_index == i].mean() for i in range(len(texts))])
        return prob, prob


class _CalibratedScoreDetector(Detector):
    """
    Zero-shot score; training labels only fit a non-decreasing calibration
    map, so the method's published direction (higher score = AI) is kept even
    when the data says otherwise.
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )

    @abstractmethod
    def _score(self, texts: List[str]) -> np.ndarray:
        pass

    def fit(self, texts, labels):
        self.calibrator.fit(self._score(texts), labels)

    def predict(self, texts):
        score = self._score(texts)
        return score, self.calibrator.predict(score)


class ZeroShotDetector(_CalibratedScoreDetector):
    # AI text tends to have higher likelihood, lower rank and lower entropy
    SIGN = {"log_likelihood": 1.0, "log_rank": -1.0, "entropy": -1.0, "fast_detectgpt": 1.0}

    def __init__(
        self,
        feature: str,
        model_id: str = DEFAULT_LM,
        max_tokens: int = 512,
        seed: int = 0,
        cache_dir: Path = Path("data/.cache/bench"),
    ):
        super().__init__()
        if feature not in LM_FEATURES:
            raise ValueError(f"feature must be one of {LM_FEATURES}, got {feature!r}")
        self.column = LM_FEATURES.index(feature)
        self.sign = self.SIGN[feature]
        self.scorer = get_lm_scorer(model_id, max_tokens, Path(cache_dir))

    def _score(self, texts):
        return self.sign * self.scorer.features(texts)[:, self.column]


class BinocularsDetector(_CalibratedScoreDetector):
    def __init__(
        self,
        observer_id: str = DEFAULT_LM,
        performer_id: str = DEFAULT_LM_INSTRUCT,
        max_tokens: int = 512,
        seed: int = 0,
        cache_dir: Path = Path("data/.cache/bench"),
    ):
        super().__init__()
        self.scorer = get_binoculars(observer_id, performer_id, max_tokens, Path(cache_dir))

    def _score(self, texts):
        # Lower Binoculars score means more likely AI
        return -self.scorer.scores(texts)


DETECTOR_TYPES: Dict[str, Type[Detector]] = {
    "tfidf": TfidfDetector,
    "embedding_lr": EmbeddingLRDetector,
    "contrastive": ContrastiveDetector,
    "zeroshot": ZeroShotDetector,
    "binoculars": BinocularsDetector,
}


def build_detector(spec: dict, seed: int, cache_dir: Path) -> Detector:
    params = {k: v for k, v in spec.items() if k not in ("name", "type")}
    try:
        cls = DETECTOR_TYPES[spec["type"]]
    except KeyError:
        raise ValueError(
            f"Unknown detector type {spec.get('type')!r}; choose from {sorted(DETECTOR_TYPES)}"
        )
    return cls(seed=seed, cache_dir=cache_dir, **params)
