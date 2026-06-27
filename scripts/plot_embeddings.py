"""2-D t-SNE of cached CLS embeddings, colored by class — for the presentation.

Shows how linearly separable each model's frozen feature space is on a benchmark
(a strong "why distillation works" slide: the student's space vs the teachers').
One panel per model. Reads eva embedding manifests (embeddings + target).

    python scripts/plot_embeddings.py --out presentation/bach_tsne.png \
        --title "BACH embeddings (t-SNE)" \
        "Student:data/embeddings_morning_1414/patho-distill/bach" \
        "UNI2-h:data/embeddings_teacher_uni2/teacher_uni2/bach" \
        "Virchow2:data/embeddings_teacher_virchow2/teacher_virchow2/bach" \
        "H-optimus-1:data/embeddings_teacher_hoptimus1/teacher_hoptimus1/bach"
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load(manifest_dir: str):
    """Return (X[N,D] float32, y[N] int) from an eva embedding manifest dir."""
    man = os.path.join(manifest_dir, "manifest.csv")
    xs, ys = [], []
    with open(man) as f:
        for row in csv.DictReader(f):
            e = torch.load(os.path.join(manifest_dir, row["embeddings"]), weights_only=False)
            e = e[0] if isinstance(e, (list, tuple)) else e
            xs.append(e.float().reshape(-1).numpy())
            ys.append(int(float(row["target"])))
    return np.stack(xs), np.asarray(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help='each "Label:manifest_dir"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Embeddings (t-SNE)")
    ap.add_argument("--classes", default=None, help="comma-sep class names for the legend")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    panels = [m.split(":", 1) for m in args.models]
    ncol = min(len(panels), 2)
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.4 * nrow), squeeze=False)
    cmap = plt.get_cmap("tab10")

    class_names = args.classes.split(",") if args.classes else None
    for i, (label, mdir) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        X, y = load(mdir)
        # PCA pre-reduction stabilizes + speeds up t-SNE on high-dim features.
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=args.seed).fit_transform(X)
        perp = min(30, max(5, len(X) // 20))
        Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=args.seed).fit_transform(X)
        for c in np.unique(y):
            m = y == c
            name = class_names[c] if class_names and c < len(class_names) else f"class {c}"
            ax.scatter(Z[m, 0], Z[m, 1], s=14, color=cmap(c % 10), label=name, alpha=0.8, linewidths=0)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    # one shared legend
    axes[0][0].legend(loc="best", fontsize=9, framealpha=0.9)
    for j in range(len(panels), nrow * ncol):  # hide empty panels
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(args.title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}  ({nrow}x{ncol} panels)")


if __name__ == "__main__":
    main()
