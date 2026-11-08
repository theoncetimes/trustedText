"""
Run a benchmark experiment described by a TOML config:

    python -m trustedtext.bench.runner configs/supervised_cv.toml

Every detector is evaluated on identical document-level splits for every
seed. Results go to results/<name>/<timestamp>/:
    predictions.csv  one row per (detector, seed, split, test document)
    summary.csv      metrics per detector and evaluation unit
    per_group.csv    accuracy per AI generator / human outlet
    summary.md       the same tables, human-readable
    run_info.json    git commit, package versions, corpus size
"""

import argparse
import csv
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from trustedtext.bench import splits as splitting
from trustedtext.bench.data import load_corpus
from trustedtext.bench.detectors import build_detector
from trustedtext.bench.metrics import (
    METRICS,
    DocumentBootstrap,
    metric_fns,
    paired_p_value,
    percentile_ci,
)
from trustedtext.bench.resources import device

log = logging.getLogger("trustedtext.bench")

CI_METRICS = ("auroc", "macro_f1")


def make_splits(split_cfg: dict, labels, groups, seed: int, n_train: int) -> List[splitting.Split]:
    kind = split_cfg["type"]
    if kind == "kfold":
        return splitting.stratified_kfold(labels, split_cfg.get("n_splits", 5), seed)
    if kind == "logo":
        return splitting.leave_one_group_out(
            labels,
            groups,
            side=split_cfg.get("side", "ai"),
            other_test_fraction=split_cfg.get("other_test_fraction", 0.2),
            seed=seed,
            min_group_size=split_cfg.get("min_group_size", 1),
        )
    if kind == "holdout":
        return splitting.holdout(n_train, len(labels))
    raise ValueError(f"Unknown split type {kind!r}; choose kfold, logo or holdout")


def run_predictions(cfg: dict, specs: List[dict], cache_dir: Path) -> List[dict]:
    data_cfg = cfg.get("data", {})
    clean = data_cfg.get("clean", True)
    docs = load_corpus(Path(data_cfg.get("dir", "data")), clean=clean)
    n_train = len(docs)
    if cfg["split"]["type"] == "holdout":
        docs += load_corpus(Path(data_cfg.get("test_dir", "data/test")), clean=clean)

    texts = [d.text for d in docs]
    labels = np.array([d.label for d in docs])
    groups = np.array([d.group for d in docs])
    log.info(f"Corpus: {int((labels == 0).sum())} human, {int((labels == 1).sum())} AI documents")

    per_unit = cfg["split"]["type"] == "logo"
    rows = []
    for seed in cfg.get("seeds", [0]):
        for split in make_splits(cfg["split"], labels, groups, seed, n_train):
            train_texts = [texts[i] for i in split.train]
            test_texts = [texts[i] for i in split.test]
            for spec in specs:
                start = time.perf_counter()
                detector = build_detector(spec, seed, cache_dir)
                detector.fit(train_texts, labels[split.train])
                score, prob = detector.predict(test_texts)
                log.info(
                    f"seed={seed} {split.name} {spec['name']}: "
                    f"{len(split.train)} train / {len(split.test)} test docs "
                    f"in {time.perf_counter() - start:.1f}s"
                )
                for i, s, p in zip(split.test, score, prob):
                    rows.append(
                        {
                            "detector": spec["name"],
                            "seed": seed,
                            "split": split.name,
                            "unit": split.name if per_unit else "all",
                            "doc_id": docs[i].doc_id,
                            "group": docs[i].group,
                            "label": int(labels[i]),
                            "score": float(s),
                            "prob": float(p),
                        }
                    )
    return rows


class _Arrays:
    """Rows of one (detector, unit), sorted so detectors align row by row."""

    def __init__(self, rows: List[dict]):
        rows = sorted(rows, key=lambda r: (r["seed"], r["split"], r["doc_id"]))
        self.keys = [(r["seed"], r["split"], r["doc_id"]) for r in rows]
        self.y = np.array([r["label"] for r in rows])
        self.score = np.array([r["score"] for r in rows])
        self.prob = np.array([r["prob"] for r in rows])
        self.seed = np.array([r["seed"] for r in rows])
        self.doc = np.array([r["doc_id"] for r in rows])

    def seed_mean(self, metric: str, idx: Optional[np.ndarray] = None) -> float:
        """Metric computed separately per seed, then averaged."""
        fn = metric_fns()[metric]
        idx = np.arange(len(self.y)) if idx is None else idx
        values = []
        for s in np.unique(self.seed[idx]):
            sub = idx[self.seed[idx] == s]
            if len(np.unique(self.y[sub])) == 2:
                values.append(fn(self.y[sub], self.score[sub], self.prob[sub]))
        return float(np.mean(values)) if values else float("nan")

    def per_seed(self, metric: str) -> np.ndarray:
        return np.array(
            [self.seed_mean(metric, np.flatnonzero(self.seed == s)) for s in np.unique(self.seed)]
        )


def summarize(rows: List[dict], reference: Optional[str], n_boot: int) -> List[dict]:
    by_key: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["unit"], r["detector"])].append(r)
    detectors = list(dict.fromkeys(r["detector"] for r in rows))
    units = list(dict.fromkeys(r["unit"] for r in rows))

    summary = []
    for unit in units:
        arrays = {d: _Arrays(by_key[(unit, d)]) for d in detectors}
        ref = arrays.get(reference)
        for name, a in arrays.items():
            out = {
                "unit": unit,
                "detector": name,
                "n_human": int(len(np.unique(a.doc[a.y == 0]))),
                "n_ai": int(len(np.unique(a.doc[a.y == 1]))),
                "n_seeds": int(len(np.unique(a.seed))),
            }
            for metric in METRICS:
                values = a.per_seed(metric)
                out[f"{metric}_mean"] = float(np.nanmean(values))
                out[f"{metric}_std"] = float(np.nanstd(values))

            boot = DocumentBootstrap(a.y, a.doc, n_boot)
            for metric in CI_METRICS:
                lo, hi = percentile_ci(boot.samples(lambda idx: a.seed_mean(metric, idx)))
                out[f"{metric}_ci_low"], out[f"{metric}_ci_high"] = lo, hi

            if ref is not None and name != reference:
                assert ref.keys == a.keys, "detectors were evaluated on different rows"
                deltas = DocumentBootstrap(a.y, a.doc, n_boot).samples(
                    lambda idx: a.seed_mean("auroc", idx) - ref.seed_mean("auroc", idx)
                )
                out["delta_auroc_vs_ref"] = out["auroc_mean"] - ref.seed_mean("auroc")
                out["p_vs_ref"] = paired_p_value(deltas)
            summary.append(out)

    if len(units) > 1:
        for name in detectors:
            unit_rows = [s for s in summary if s["detector"] == name]
            macro = {"unit": "macro", "detector": name}
            for metric in METRICS:
                macro[f"{metric}_mean"] = float(np.nanmean([s[f"{metric}_mean"] for s in unit_rows]))
            summary.append(macro)
    return summary


def per_group(rows: List[dict]) -> List[dict]:
    """Accuracy at P(AI)=0.5 per generator/outlet: detection rate for AI, 1-FPR for human."""
    buckets: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["detector"], r["label"], r["group"])].append(r)
    out = []
    for (detector, label, group), rs in sorted(buckets.items()):
        prob = np.array([r["prob"] for r in rs])
        correct = (prob >= 0.5).astype(int) == label
        out.append(
            {
                "detector": detector,
                "class": "ai" if label == 1 else "human",
                "group": group,
                "n_docs": len({r["doc_id"] for r in rs}),
                "accuracy": float(correct.mean()),
                "mean_prob_ai": float(prob.mean()),
            }
        )
    return out


def _fmt(value, ci=None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    text = f"{value:.3f}"
    if ci is not None:
        text += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return text


def render_markdown(cfg: dict, summary: List[dict], reference: Optional[str]) -> str:
    lines = [f"# {cfg['name']}", ""]
    if cfg.get("description"):
        lines += [cfg["description"].strip(), ""]
    lines += [
        "Mean over seeds; 95% CIs from a document-level stratified bootstrap. "
        f"Δ and p compare AUROC against `{reference}` (paired bootstrap).",
        "",
    ]
    units = list(dict.fromkeys(s["unit"] for s in summary))
    detail_units = [u for u in units if u != "macro"]

    if len(detail_units) > 1:
        detectors = list(dict.fromkeys(s["detector"] for s in summary))
        lookup = {(s["unit"], s["detector"]): s for s in summary}
        lines += ["## AUROC per held-out group", ""]
        lines.append("| Detector | " + " | ".join(units) + " |")
        lines.append("|---" * (len(units) + 1) + "|")
        for d in detectors:
            cells = [_fmt(lookup[(u, d)]["auroc_mean"]) for u in units]
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")

    for unit in detail_units:
        rows = [s for s in summary if s["unit"] == unit]
        first = rows[0]
        lines += [f"## {unit} ({first['n_human']} human, {first['n_ai']} AI documents)", ""]
        lines.append(
            "| Detector | AUROC [95% CI] | TPR@1%FPR | TPR@5%FPR | Macro-F1 [95% CI] | ECE | ΔAUROC (p) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for s in rows:
            delta = (
                f"{s['delta_auroc_vs_ref']:+.3f} ({s['p_vs_ref']:.3f})"
                if "p_vs_ref" in s
                else "ref" if s["detector"] == reference else "–"
            )
            lines.append(
                f"| {s['detector']} "
                f"| {_fmt(s['auroc_mean'], (s['auroc_ci_low'], s['auroc_ci_high']))} "
                f"| {_fmt(s['tpr@1%fpr_mean'])} | {_fmt(s['tpr@5%fpr_mean'])} "
                f"| {_fmt(s['macro_f1_mean'], (s['macro_f1_ci_low'], s['macro_f1_ci_high']))} "
                f"| {_fmt(s['ece_mean'])} | {delta} |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_info(n_rows: int, elapsed: float) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device(),
        "packages": {
            p: version(p) for p in ("torch", "transformers", "sentence-transformers", "scikit-learn", "numpy")
        },
        "n_prediction_rows": n_rows,
        "elapsed_seconds": round(elapsed, 1),
    }


def run_experiment(config_path: Path, output_root: Path, only: Optional[List[str]] = None) -> Path:
    cfg = tomllib.loads(config_path.read_text())
    specs = cfg["detectors"]
    if only:
        specs = [s for s in specs if s["name"] in only]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Detector names must be unique: {names}")

    eval_cfg = cfg.get("evaluation", {})
    reference = eval_cfg.get("reference")
    if reference not in names:
        reference = None
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "data/.cache/bench"))

    start = time.perf_counter()
    rows = run_predictions(cfg, specs, cache_dir)
    summary = summarize(rows, reference, eval_cfg.get("n_bootstrap", 1000))

    out_dir = output_root / cfg["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.toml")
    _write_csv(out_dir / "predictions.csv", rows)
    _write_csv(out_dir / "summary.csv", summary)
    _write_csv(out_dir / "per_group.csv", per_group(rows))
    report = render_markdown(cfg, summary, reference)
    (out_dir / "summary.md").write_text(report + "\n")
    (out_dir / "run_info.json").write_text(
        json.dumps(_run_info(len(rows), time.perf_counter() - start), indent=2) + "\n"
    )

    print(report)
    print(f"Results written to {out_dir}")
    return out_dir


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a TrustedText benchmark experiment")
    parser.add_argument("config", type=Path, help="Experiment TOML file (see configs/)")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--only", help="Comma-separated detector names to run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The training pipeline logs every epoch; too noisy across many folds
    logging.getLogger("trustedtext.pipeline").setLevel(logging.WARNING)

    only = args.only.split(",") if args.only else None
    run_experiment(args.config, args.output_root, only)


if __name__ == "__main__":
    sys.exit(main())
