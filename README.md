# TrustedText

A research codebase for detecting machine-generated text. It contains:

- **A detector**: a small MLP trained with cross-entropy plus supervised
  contrastive loss on [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  chunk embeddings (1024-d), with a FastAPI demo.
- **A benchmark harness** (`trustedtext.bench`) that evaluates that detector
  against standard supervised and zero-shot baselines under document-level
  splits, with confidence intervals and paired significance tests.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

## Status

The current corpus is small (42 human and 36 AI news articles, plus an
11-document test set) and is not distributed with the repository. Results on
it have wide confidence intervals and should be read as a pilot, not as
evidence that any detector works in general. See
[Limitations](#limitations).

## Installation

```bash
uv sync                # or: pip install -e . && pip install pytest
uv run pytest          # unit tests for splits, metrics and data loading
```

## Data

One document per UTF-8 `.txt` file. The filename prefix records the source,
which the benchmark uses for per-source analysis and leave-one-group-out
splits: `claude003.txt` belongs to generator `claude`, `reuters001.txt` to
outlet `reuters`.

```
data/
├── human/   # human-written documents, e.g. reuters001.txt
├── ai/      # machine-generated documents, e.g. gpt001.txt
└── test/    # optional fixed held-out set with the same structure
    ├── human/
    └── ai/
```

By default the benchmark strips artifacts that reveal where a text was copied
from rather than how it was written (zero-width characters, markdown
emphasis, headings and links); set `clean = false` in a config to disable this.

Tools for collecting data live in `scripts/` (`add_data.py` for pasting
samples, `generate_content.py` for generating AI text through OpenRouter).

## Running experiments

Each experiment is a TOML file in `configs/`:

| Config | Question |
|---|---|
| `quick.toml` | Smoke test with TF-IDF baselines (seconds) |
| `supervised_cv.toml` | In-distribution comparison of supervised detectors |
| `zeroshot_cv.toml` | Zero-shot detectors: likelihood, log-rank, entropy, Fast-DetectGPT, Binoculars |
| `leave_one_generator_out.toml` | Generalization to an unseen AI generator |
| `leave_one_outlet_out.toml` | False positives on an unseen human source |
| `ablation_contrastive.toml` | Which components of the TrustedText classifier matter |
| `holdout_test.toml` | Train on `data/`, evaluate on `data/test/` |

```bash
uv run trustedtext-bench configs/supervised_cv.toml
uv run trustedtext-bench configs/zeroshot_cv.toml --only fast_detectgpt,binoculars
```

Each run writes to `results/<name>/<timestamp>/`:

- `summary.md` / `summary.csv`: metrics per detector
- `per_group.csv`: accuracy per generator and per outlet
- `predictions.csv`: every test prediction, for re-analysis
- `config.toml`, `run_info.json`: exact config, git commit and package versions

### Evaluation protocol

- **Document-level splits.** Long documents are split into chunks for the
  embedding-based detectors, but every split assigns whole documents, so
  chunks of one document never appear on both sides.
- **Repeated seeds.** Splits and model initialization are repeated over the
  configured seeds; metrics are averaged over seeds.
- **Metrics.** AUROC; TPR at 1% and 5% FPR (the operating points that matter
  when false accusations are costly); macro-F1 and accuracy at P(AI) = 0.5;
  expected calibration error.
- **Uncertainty.** 95% confidence intervals from a stratified bootstrap over
  documents (all seeds of a document are resampled together). Each detector is
  compared with a reference detector by a paired bootstrap test on AUROC.
- **No tuning on test data.** Linear-model regularization is chosen by inner
  cross-validation on the training fold. Zero-shot detectors use the training
  fold only to calibrate P(AI); their AUROC does not depend on training data.

### Detectors

| Type | Description |
|---|---|
| `tfidf` | Word or character n-gram TF-IDF + logistic regression |
| `embedding_lr` | Logistic regression on mean Qwen3 chunk embeddings |
| `contrastive` | The TrustedText classifier (`trustedtext.pipeline`) |
| `zeroshot` | Log-likelihood, log-rank, entropy, or Fast-DetectGPT ([Bao et al., 2024](https://arxiv.org/abs/2310.05130)) under one causal LM |
| `binoculars` | Binoculars ([Hans et al., 2024](https://arxiv.org/abs/2401.12070)) with an observer/performer model pair |

Zero-shot detectors default to Qwen2.5-0.5B (base, plus Instruct as the
Binoculars performer). The original papers use 2.7B–7B models, so these
numbers are a lower bound on those methods.

To add a detector, subclass `Detector` in `src/trustedtext/bench/detectors.py`,
implement `fit` and `predict`, and register it in `DETECTOR_TYPES`.

## Training and serving the TrustedText model

```bash
uv run trustedtext-train               # trains on data/, evaluates on data/test/, saves models/
uv run trustedtext-train --visualize   # also plots raw vs. projected embeddings
uv run python app/api.py               # web UI and API on http://localhost:8000
```

```python
from trustedtext import TrustedText

detector = TrustedText()
detector.load()
label, ai_probability = detector.predict("Your text here")
```

Long inputs are split into chunks and the AI probability is averaged over
chunks, both in the library and in the API.

Environment variables `MAX_SEQ_LEN` (default 320) and `EMBED_BATCH_SIZE`
(default 4) control embedding for training.

## Project structure

```
src/trustedtext/
├── pipeline.py        # TrustedText detector: embedding, contrastive classifier, training
├── train.py           # training CLI
└── bench/
    ├── data.py        # corpus loading, source metadata, artifact cleaning
    ├── splits.py      # k-fold, leave-one-group-out, holdout
    ├── detectors.py   # detector interface and implementations
    ├── resources.py   # cached embedding model and causal-LM scorers
    ├── metrics.py     # AUROC, TPR@FPR, calibration, document bootstrap
    └── runner.py      # experiment CLI
configs/               # experiment definitions
tests/                 # unit tests
app/                   # FastAPI server and web UI
scripts/               # data collection utilities
```

## Limitations

- **Small corpus.** With about 80 documents, a single misclassified document
  moves accuracy by several points; rely on the confidence intervals.
- **Narrow domain.** Human texts are professional news articles and AI texts
  are LLM-generated news, so results may not transfer to essays, chat,
  social media or other languages.
- **Source confounds.** Human and AI documents differ in topic, outlet and
  formatting, so detectors may learn those instead of authorship. Topic-matched
  generation (prompting each model with a human article's headline and key
  facts) would remove most of this.
- **Label purity.** Recent human-published news may itself be AI-assisted;
  documents published before November 2022 are a safer human baseline.

## License

MIT; see [LICENSE](LICENSE). Check the terms of the sources you collect text
from before redistributing any data.
