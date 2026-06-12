#!/usr/bin/env python3
"""Run the Kaiko `eva` linear-probe benchmark on our distilled student.

For each dataset, this invokes eva's own (maintained) dataset config and only
swaps in our backbone via `configs/eva/student_backbone.yaml`. eva extracts
frozen CLS embeddings, trains a probe head, and reports leaderboard-comparable
metrics (balanced accuracy / accuracy).

    python benchmark/run_eva.py --checkpoint checkpoints/last.ckpt \
        --datasets bach mhist --data-root ./data --download

Prereqs (on the VM): `pip install kaiko-eva` (or `pip install -e .[eval]`) and
this repo importable so `distill.eval.student_backbone` resolves.

NOTE: eva's exact CLI/config schema is validated on the VM where eva is
installed — the wrapper + override here are written against eva's documented
`ModelFromFunction` interface but have not been run end-to-end yet.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERRIDE_CONFIG = REPO_ROOT / "configs" / "eva" / "student_backbone.yaml"

# Pin to a known eva ref for reproducibility; override with --eva-ref or point
# --eva-config-dir at a local clone of the eva repo.
EVA_REF = "main"
EVA_CONFIG_BASE = "configs/vision/pathology/offline/classification"

# dataset -> (eva config filename, whether eva can auto-download it)
DATASETS = {
    "bach": ("bach.yaml", True),
    "mhist": ("mhist.yaml", False),          # manual download (license form)
    "camelyon16": ("camelyon16.yaml", False),  # large, manual download
    "panda_small": ("panda_small.yaml", False),  # large, manual download
}

STUDENT_IN_FEATURES = "768"  # student CLS embedding dim


def eva_config_ref(name: str, eva_ref: str, eva_config_dir: str | None) -> str:
    """Local path (if a clone is given) else a pinned raw GitHub URL."""
    filename, _ = DATASETS[name]
    if eva_config_dir:
        return str(Path(eva_config_dir) / EVA_CONFIG_BASE / filename)
    return (
        f"https://raw.githubusercontent.com/kaiko-ai/eva/{eva_ref}/"
        f"{EVA_CONFIG_BASE}/{filename}"
    )


def build_env(args, dataset: str) -> dict:
    env = os.environ.copy()
    env["MODEL_NAME"] = args.model_name
    env["IN_FEATURES"] = STUDENT_IN_FEATURES
    if args.checkpoint:
        env["CHECKPOINT_PATH"] = str(Path(args.checkpoint).resolve())
    env["DATA_ROOT"] = str(Path(args.data_root).resolve() / dataset)
    env["EMBEDDINGS_ROOT"] = str(Path(args.embeddings_root).resolve() / dataset)
    env["OUTPUT_ROOT"] = str(Path(args.output_root).resolve() / args.model_name / dataset)
    env["DOWNLOAD_DATA"] = "true" if args.download else "false"
    # Make `distill` importable inside the eva process.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def run_one(dataset: str, args) -> int:
    _, can_download = DATASETS[dataset]
    if args.download and not can_download:
        print(
            f"[{dataset}] WARNING: not auto-downloadable; provide it under "
            f"{Path(args.data_root) / dataset} manually.",
            file=sys.stderr,
        )
    cfg = eva_config_ref(dataset, args.eva_ref, args.eva_config_dir)
    cmd = [
        "eva", args.subcommand,
        "--config", cfg,
        "--config", str(OVERRIDE_CONFIG),
    ]
    env = build_env(args, dataset)
    print(f"\n=== [{dataset}] {' '.join(cmd)}")
    print(f"    OUTPUT_ROOT={env['OUTPUT_ROOT']}  CHECKPOINT_PATH={env.get('CHECKPOINT_PATH','<none>')}")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


def collect_results(args) -> dict:
    """Best-effort scrape of eva's results JSON per dataset."""
    summary = {}
    for dataset in args.datasets:
        out_dir = Path(args.output_root).resolve() / args.model_name / dataset
        hits = sorted(out_dir.rglob("results.json")) if out_dir.exists() else []
        if not hits:
            summary[dataset] = None
            continue
        try:
            summary[dataset] = json.loads(hits[-1].read_text())
        except Exception as e:  # noqa: BLE001 — best-effort reporting only
            summary[dataset] = {"error": str(e)}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="Path to a LightningModel .ckpt (omit for raw DINOv2).")
    p.add_argument("--datasets", nargs="+", default=["bach", "mhist"],
                   choices=list(DATASETS), help="Datasets to benchmark.")
    p.add_argument("--data-root", default="./data", help="Parent dir holding <dataset>/ data.")
    p.add_argument("--embeddings-root", default="./data/embeddings",
                   help="Where eva caches extracted embeddings.")
    p.add_argument("--output-root", default="./logs", help="Where eva writes results.")
    p.add_argument("--model-name", default="patho-distill", help="Label for output paths.")
    p.add_argument("--download", action="store_true",
                   help="Ask eva to auto-download (only BACH supports this).")
    p.add_argument("--subcommand", default="predict_fit",
                   choices=["predict_fit", "fit", "predict"],
                   help="eva subcommand (predict_fit = extract embeddings then probe).")
    p.add_argument("--eva-ref", default=EVA_REF, help="eva git ref for raw-config URLs.")
    p.add_argument("--eva-config-dir", default=os.environ.get("EVA_CONFIG_DIR"),
                   help="Local clone of the eva repo (skips raw-URL fetch).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = p.parse_args()

    failures = [d for d in args.datasets if run_one(d, args) != 0]

    print("\n===== Summary =====")
    results = collect_results(args)
    for dataset in args.datasets:
        status = "FAILED" if dataset in failures else "ok"
        print(f"  {dataset:12s} [{status}]  results: {results.get(dataset)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
