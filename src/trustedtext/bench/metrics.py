"""
Detection metrics and document-level bootstrap.

Throughout, `score` is a real number where higher means "more likely AI"
(used for threshold-free metrics), and `prob` is a calibrated P(AI)
(used for thresholded and calibration metrics, with threshold 0.5).
"""

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

METRICS = ("auroc", "tpr@1%fpr", "tpr@5%fpr", "macro_f1", "accuracy", "ece")


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count half); much faster than sklearn inside bootstraps."""
    ranks = rankdata(score)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= max_fpr].max())


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - prob[in_bin].mean())
    return float(ece)


def metric_fns() -> Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]]:
    return {
        "auroc": lambda y, s, p: auroc(y, s),
        "tpr@1%fpr": lambda y, s, p: tpr_at_fpr(y, s, 0.01),
        "tpr@5%fpr": lambda y, s, p: tpr_at_fpr(y, s, 0.05),
        "macro_f1": lambda y, s, p: macro_f1(y, (p >= 0.5).astype(int)),
        "accuracy": lambda y, s, p: float(np.mean((p >= 0.5).astype(int) == y)),
        "ece": lambda y, s, p: expected_calibration_error(y, p),
    }


def compute_metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    return {name: fn(y, score, prob) for name, fn in metric_fns().items()}


class DocumentBootstrap:
    """
    Stratified bootstrap that resamples documents, not prediction rows.

    A document can have several rows (one per seed); all of them are kept
    together so repeated seeds don't masquerade as independent samples.
    """

    def __init__(self, y: np.ndarray, doc_ids: np.ndarray, n_boot: int, seed: int = 0):
        self.n_boot = n_boot
        self.rng = np.random.default_rng(seed)
        docs, inverse = np.unique(doc_ids, return_inverse=True)
        self.rows_by_doc = [np.flatnonzero(inverse == k) for k in range(len(docs))]
        doc_labels = np.array([y[rows[0]] for rows in self.rows_by_doc])
        self.strata = [np.flatnonzero(doc_labels == c) for c in np.unique(doc_labels)]

    def samples(self, statistic: Callable[[np.ndarray], float]) -> np.ndarray:
        """Evaluate statistic(row_indices) on n_boot resamples."""
        out = np.empty(self.n_boot)
        for b in range(self.n_boot):
            picked = np.concatenate([self.rng.choice(s, size=len(s)) for s in self.strata])
            out[b] = statistic(np.concatenate([self.rows_by_doc[k] for k in picked]))
        return out


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_p_value(delta_samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0."""
    p = 2 * min(np.mean(delta_samples <= 0), np.mean(delta_samples >= 0))
    return float(min(1.0, p))
