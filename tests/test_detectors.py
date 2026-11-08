import numpy as np

from trustedtext.bench.detectors import TfidfDetector, _CalibratedScoreDetector, build_detector


class _FixedScore(_CalibratedScoreDetector):
    def __init__(self, scores):
        super().__init__()
        self.scores = dict(scores)

    def _score(self, texts):
        return np.array([self.scores[t] for t in texts])


def test_calibration_never_reverses_score_direction():
    # Data where the zero-shot score points the wrong way
    texts = [f"t{i}" for i in range(6)]
    det = _FixedScore(zip(texts, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]))
    det.fit(texts, np.array([0, 0, 0, 1, 1, 1]))
    score, prob = det.predict(texts)
    assert np.all(np.diff(prob[np.argsort(score)]) >= 0)


def test_tfidf_detector_separates_toy_data():
    texts = ["the cat sat"] * 5 + ["quantum flux capacitor"] * 5
    labels = np.array([0] * 5 + [1] * 5)
    det = build_detector({"name": "t", "type": "tfidf", "C": 10.0}, seed=0, cache_dir=None)
    assert isinstance(det, TfidfDetector)
    det.fit(texts, labels)
    score, prob = det.predict(["the cat sat", "quantum flux capacitor"])
    assert prob[0] < 0.5 < prob[1]
    assert np.array_equal(score, prob)
