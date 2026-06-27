"""Add 5-fold cross-validation columns to an eva embeddings manifest.

This is the data-prep step for running eva's OWN fit code as cross-validation,
exactly the scheme the prof described: one manifest where each `fold{k}` column
assigns every sample to `train`/`val`, and **each sample is `val` in exactly one
fold** (a clean stratified 5-fold partition, ~20% held out per fold). eva reads a
chosen fold column as its split via the dataset `column_mapping`
(`{"split": "fold{k}"}`), so running `eva fit` once per fold = 5-fold CV with the
head retrained from scratch each time (eva clones + re-inits the head per run).

The original designated `split` column (BACH 268/132, MHIST's official 2175/977)
is left untouched in the output, so it can be run as a SEPARATE single fit
(`FOLD_COLUMN=split`) to report the literature/Table-1-comparable number next to
the CV -- it is NOT one of the folds.

The frozen backbone makes embeddings deterministic, so embed ONCE (eva predict),
then CV the cheap linear head over the pooled samples.

Input: an eva embeddings `manifest.csv` (cols incl. `embeddings`, `target`,
`split`). Output: the same rows plus `fold0..fold{n-1}`, written next to it.

    python scripts/make_fold_manifest.py \
        --manifest data/embeddings_.../bach/manifest.csv \
        --out      data/embeddings_.../bach/manifest_cv.csv

Stratification mirrors sklearn StratifiedKFold(shuffle=True, random_state=42),
matching the seed used elsewhere, so folds are class-balanced and reproducible.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def add_fold_columns(
    df: pd.DataFrame, target_col: str, folds: int, seed: int
) -> pd.DataFrame:
    """Return `df` with `fold0..fold{folds-1}` columns of 'train'/'val'.

    Clean stratified k-fold PARTITION (the prof's scheme): each sample is 'val' in
    exactly one fold (~1/folds held out per fold), class-balanced and reproducible
    (StratifiedKFold(shuffle=True, random_state=seed)). The original designated
    `split` column is left intact for a separate single fit.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y = df[target_col].astype(int).to_numpy()
    for k in range(folds):
        df[f"fold{k}"] = "train"
    for k, (_, val_idx) in enumerate(skf.split(df.index.to_numpy(), y)):
        df.iloc[val_idx, df.columns.get_loc(f"fold{k}")] = "val"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to eva manifest.csv")
    ap.add_argument("--out", required=True, help="path for the augmented manifest")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.target_col not in df.columns:
        raise KeyError(
            f"target column {args.target_col!r} not in manifest "
            f"(columns: {list(df.columns)})"
        )
    df = add_fold_columns(df, args.target_col, args.folds, args.seed)

    # Sanity: every sample is held out in exactly one fold (clean partition).
    held_out = (df[[f"fold{k}" for k in range(args.folds)]] == "val").sum(axis=1)
    assert (held_out == 1).all(), "each sample must be 'val' in exactly one fold"

    df.to_csv(args.out, index=False)
    val_sizes = {f"fold{k}": int((df[f"fold{k}"] == "val").sum()) for k in range(args.folds)}
    print(f"wrote {args.out}: N={len(df)} folds={args.folds} (clean partition) val-sizes={val_sizes}")
    # Per-class balance of the held-out folds (BACH is tiny + imbalanced, so worth seeing).
    for k in range(args.folds):
        vc = df.loc[df[f"fold{k}"] == "val", args.target_col].value_counts().sort_index()
        print(f"  fold{k} val class counts: {vc.to_dict()}")


if __name__ == "__main__":
    main()
