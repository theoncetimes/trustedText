import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from trustedtext.bench.metrics import (
    DocumentBootstrap,
    auroc,
    compute_metrics,
    expected_calibration_error,
    macro_f1,
    paired_p_value,
    percentile_ci,
    tpr_at_fpr,
)


def test_auroc_and_macro_f1_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = rng.normal(size=200) + y
    score[:20] = 0.5  # include ties
    pred = (score > 0.5).astype(int)
    assert auroc(y, score) == pytest.approx(roc_auc_score(y, score))
    assert macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro"))


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = compute_metrics(y, score, score)
    assert m["auroc"] == 1.0
    assert m["tpr@1%fpr"] == 1.0
    assert m["macro_f1"] == 1.0


def test_tpr_at_fpr_requires_zero_false_positives_at_one_percent():
    y = np.array([0] * 10 + [1] * 10)
    score = np.concatenate([np.linspace(0, 0.5, 9), [0.95], np.linspace(0.6, 0.9, 10)])
    # The best-scoring human outranks every AI text, so no threshold has FPR <= 1%
    assert tpr_at_fpr(y, score, 0.01) == 0.0
    assert tpr_at_fpr(y, score, 0.10) == 1.0


def test_calibration_error():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_bootstrap_keeps_document_rows_together():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    docs = np.array(["a", "b", "c", "d", "a", "b", "c", "d"])  # two seeds
    boot = DocumentBootstrap(y, docs, n_boot=50, seed=0)
    sizes = boot.samples(lambda rows: len(rows))
    assert np.all(sizes == 8)  # 4 docs resampled x 2 rows each
    labels_per_sample = boot.samples(lambda rows: y[rows].sum())
    assert np.all(labels_per_sample == 4)  # stratified: class balance preserved


def test_bootstrap_ci_and_paired_p():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    docs = np.arange(80)
    good = y + rng.normal(scale=0.5, size=80)
    bad = rng.normal(size=80)
    boot = DocumentBootstrap(y, docs, n_boot=300, seed=0)
    lo, hi = percentile_ci(boot.samples(lambda i: auroc(y[i], good[i])))
    assert lo <= auroc(y, good) <= hi
    deltas = DocumentBootstrap(y, docs, n_boot=300, seed=0).samples(
        lambda i: auroc(y[i], good[i]) - auroc(y[i], bad[i])
    )
    assert paired_p_value(deltas) < 0.01
