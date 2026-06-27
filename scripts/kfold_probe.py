"""5-fold cross-validation linear probe on cached CLS embeddings (BACH / MHIST).

The frozen backbone's CLS embedding for every sample is cached as one ``.pt``
file (a 1-element list holding the embedding tensor) next to a ``manifest.csv``
(cols: ``origin,embeddings,target,split``; ``embeddings`` = relative .pt path,
``target`` = class int). Because the backbone is frozen, these embeddings are
deterministic, so cross-validation only needs to re-fit the cheap linear head.

Default mode (``--kfold``) pools ALL manifest rows (ignoring the ``split``
column), runs ``StratifiedKFold(n_splits, shuffle=True, random_state=seed)``,
and for each fold fits a fresh linear head and evaluates the held-out fold.
Each fold therefore differs in BOTH the data partition AND the head init —
i.e. genuine fold variance, not the head-init noise eva's ``n_runs`` measured.

``--fixed-split`` instead trains on the manifest's ``train`` rows and evaluates
its ``val`` rows (single split), reproducing eva's protocol — used to sanity
check that a given embedding dir maps to the reported number.

Head recipe mirrors eva 0.4.5 for comparability (configs/eva/bach_offline_*.yaml):
``nn.Linear(dim, n_classes)``, ``AdamW(lr=3e-4)``, ``CrossEntropyLoss``, batch 256,
``max_steps=12500``, and -- crucially -- ``checkpoint_type: best``: eva monitors
the eval-set accuracy and reports the BEST checkpoint, not the final step. So this
script evaluates periodically and reports the peak eval accuracy (and the AUROC at
that same checkpoint).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(manifest: str) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Return (X[N, D] float, y[N] int, split[N])."""
    root = os.path.dirname(manifest)
    xs, ys, splits = [], [], []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            emb = torch.load(os.path.join(root, row["embeddings"]), weights_only=False)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            xs.append(emb.float().reshape(-1))
            ys.append(int(float(row["target"])))
            splits.append(row["split"])
    X = torch.stack(xs)
    return X, np.asarray(ys, dtype=np.int64), splits


def train_eval(
    X_tr: torch.Tensor,
    y_tr: np.ndarray,
    X_te: torch.Tensor,
    y_te: np.ndarray,
    n_classes: int,
    device: str,
    steps: int,
    lr: float,
    batch: int,
    seed: int,
    select: str = "best",
) -> tuple[float, float]:
    """Fit a fresh linear head; return (accuracy, AUROC) on the held-out set.

    ``select="best"`` mirrors eva's ``checkpoint_type: best``: track the eval-set
    accuracy across training and keep the peak (and the AUROC at that same
    checkpoint). In k-fold this peeks at the held-out fold for selection.

    ``select="final"`` is the no-peek CV protocol: train to convergence and report
    the held-out fold once, at the final step (no selection on the test fold).
    """
    torch.manual_seed(seed)
    dim = X_tr.shape[1]
    head = nn.Linear(dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    X_tr = X_tr.to(device)
    X_te = X_te.to(device)
    yt = torch.from_numpy(y_tr).to(device)
    n = X_tr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def eval_once() -> tuple[float, float]:
        head.eval()
        with torch.no_grad():
            logits = head(X_te)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
        head.train()
        # eva's MulticlassAccuracy uses macro averaging (== balanced accuracy).
        a = float(balanced_accuracy_score(y_te, preds))
        if n_classes == 2:
            au = float(roc_auc_score(y_te, probs[:, 1]))
        else:
            au = float(roc_auc_score(y_te, probs, multi_class="ovr", average="macro"))
        return a, au

    # eva validates once per epoch (Lightning default) and keeps the best
    # checkpoint, so mirror that evaluation cadence.
    eval_every = max(1, n // batch)
    best_acc, best_auroc = eval_once()
    last_acc, last_auroc = best_acc, best_auroc
    head.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
        opt.zero_grad()
        loss = crit(head(X_tr[idx]), yt[idx])
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            last_acc, last_auroc = eval_once()
            if last_acc > best_acc:
                best_acc, best_auroc = last_acc, last_auroc
    if select == "final":
        return last_acc, last_auroc
    return best_acc, best_auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to manifest.csv")
    ap.add_argument("--mode", choices=["kfold", "fixed-split"], default="kfold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5,
                    help="fixed-split mode: number of head-init seeds to average")
    ap.add_argument("--steps", type=int, default=12500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select", choices=["best", "final"], default="best",
                    help="best = peek at held-out fold for checkpoint (eva-style); "
                         "final = no-peek, report converged head")
    ap.add_argument("--out", default=None, help="optional path to dump JSON")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, splits = load_manifest(args.manifest)
    n_classes = int(y.max()) + 1
    log(f"{args.manifest}: N={len(y)} dim={X.shape[1]} classes={n_classes} device={device}")

    if args.mode == "fixed-split":
        tr = np.array([s == "train" for s in splits])
        te = np.array([s == "val" for s in splits])
        if te.sum() == 0:  # some manifests label the held-out split "test"
            te = np.array([s == "test" for s in splits])
        accs, aurocs = [], []
        for s in range(args.seeds):
            acc, auroc = train_eval(
                X[tr], y[tr], X[te], y[te], n_classes, device,
                args.steps, args.lr, args.batch, args.seed + s, args.select,
            )
            accs.append(acc)
            aurocs.append(auroc)
        accs, aurocs = np.array(accs), np.array(aurocs)
        log(f"FIXED-SPLIT ({args.seeds} seeds): n_train={int(tr.sum())} "
            f"n_test={int(te.sum())} acc {accs.mean()*100:.2f} +- {accs.std()*100:.2f}  "
            f"auroc {aurocs.mean()*100:.2f} +- {aurocs.std()*100:.2f}")
        res = {
            "mode": "fixed-split", "seeds": args.seeds,
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
            "seed_acc": accs.tolist(), "seed_auroc": aurocs.tolist(),
        }
    else:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        accs, aurocs = [], []
        for k, (tr_i, te_i) in enumerate(skf.split(np.zeros(len(y)), y)):
            acc, auroc = train_eval(
                X[tr_i], y[tr_i], X[te_i], y[te_i], n_classes, device,
                args.steps, args.lr, args.batch, args.seed + k, args.select,
            )
            accs.append(acc)
            aurocs.append(auroc)
            log(f"fold {k}: n_train={len(tr_i)} n_test={len(te_i)} "
                f"acc={acc*100:.2f} auroc={auroc*100:.2f}")
        accs, aurocs = np.array(accs), np.array(aurocs)
        log(f"5-FOLD: acc {accs.mean()*100:.2f} +- {accs.std()*100:.2f}  "
            f"auroc {aurocs.mean()*100:.2f} +- {aurocs.std()*100:.2f}")
        res = {
            "mode": "kfold", "folds": args.folds,
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
            "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
            "fold_acc": accs.tolist(), "fold_auroc": aurocs.tolist(),
        }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
