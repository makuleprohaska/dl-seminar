"""Aggregate per-fold eva `results.json` into a 5-fold CV mean +/- std.

Each `eva fit` fold run writes `<output>/<foldK>/results.json` with eva's metric
schema: ``{"metrics": {"val": [ {"val/MulticlassAccuracy": {"mean","stdev","values"}}, ... ]}}``.
With n_runs=1 each fold contributes a single value; this script pools the five
fold values and reports the cross-validation mean and standard deviation -- the
number that goes in the handout.

    python scripts/collect_cv.py logs/cv/pathology/patho_distill/fold*/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics


def _fold_value(path: str, metric_substr: str) -> tuple[float, str]:
    with open(path) as f:
        results = json.load(f)
    val_datasets = results.get("metrics", {}).get("val", [])
    if not val_datasets:
        raise ValueError(f"{path}: no val metrics recorded")
    metrics = val_datasets[0]
    # Metric keys are Lightning's logged names, e.g. "val/MulticlassAccuracy".
    matches = [k for k in metrics if metric_substr.lower() in k.lower()]
    if not matches:
        raise KeyError(f"{path}: no metric matching {metric_substr!r} in {list(metrics)}")
    key = sorted(matches, key=len)[0]  # prefer the plain accuracy over F1/AUROC variants
    values = metrics[key]["values"]
    return float(values[0]), key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="per-fold results.json files")
    ap.add_argument("--metric", default="Accuracy", help="substring of the val metric")
    args = ap.parse_args()

    vals, key = [], None
    for path in sorted(args.results):
        v, key = _fold_value(path, args.metric)
        vals.append(v)
        print(f"  {path}: {v * 100:.2f}")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n{key} over {len(vals)} folds: {mean * 100:.2f} +- {std * 100:.2f}")


if __name__ == "__main__":
    main()
