"""Train an ABMIL head on cached CAMELYON16 per-slide patch-embedding bags.

Slide-level MIL: each slide is a bag of patch embeddings (produced by
`camelyon16_stream_embed.py`); an attention-based MIL head (eva's `ABMIL`)
pools the bag into one slide-level logit and is trained to predict
normal(0)/tumor(1).

Protocol (mirrors Kaiko eva's CAMELYON16 offline benchmark, but a SINGLE run,
not 5): train on the 216 training slides, early-select on the 54-slide val
split (eva's fixed `_val_slides`), and report on the 129-slide official test set
(`test_*`, labels from evaluation/reference.csv, already baked into each bag).

One model per invocation (`--model`); embed dim is inferred from the bags.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from eva.vision.data.datasets.classification.camelyon16 import Camelyon16
from eva.vision.models.networks.abmil import ABMIL

VAL_SLIDES = set(Camelyon16._val_slides)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
class BagDataset(Dataset):
    """One slide-bag per item: (embeddings[N, D], label)."""

    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        d = torch.load(self.files[i])
        return d["embeddings"].float(), int(d["label"])


def split_files(model_dir: str) -> dict[str, list[str]]:
    """Partition slide bags into train / val / test by slide id."""
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.pt"))):
        sid = os.path.basename(f)[:-3]
        if sid.startswith("test"):
            splits["test"].append(f)
        elif sid in VAL_SLIDES:
            splits["val"].append(f)
        else:  # normal_* / tumor_* training slides not in the val list
            splits["train"].append(f)
    return splits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return (accuracy, AUROC) over a loader of single-slide bags."""
    model.eval()
    correct = total = 0
    ys, ps = [], []
    for bag, label in loader:
        bag = bag.to(device)  # [1, N, D]
        logits = model(bag)  # [1, 2]
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(logits.argmax(dim=1).item())
        y = int(label.item())
        correct += int(pred == y)
        total += 1
        ys.append(y)
        ps.append(prob)
    acc = correct / max(total, 1)
    auroc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return acc, auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="student | uni2 | virchow2 | hoptimus1")
    ap.add_argument("--root", default="data/camelyon16_embeddings")
    ap.add_argument("--out", default="logs/camelyon16_abmil")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--projected-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = os.path.join(args.root, args.model)
    splits = split_files(model_dir)
    log(f"{args.model}: train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])}")

    # Infer embed dim from the first bag.
    dim = torch.load(splits["train"][0])["embeddings"].shape[1]
    log(f"embed dim = {dim}")

    train_loader = DataLoader(BagDataset(splits["train"]), batch_size=1, shuffle=True)
    val_loader = DataLoader(BagDataset(splits["val"]), batch_size=1)
    test_loader = DataLoader(BagDataset(splits["test"]), batch_size=1)

    model = ABMIL(
        input_size=dim,
        output_size=2,
        projected_input_size=args.projected_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_test = (0.0, float("nan"))
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for bag, label in train_loader:
            bag = bag.to(device)
            label = label.to(device)
            opt.zero_grad()
            loss = criterion(model(bag), label)
            loss.backward()
            opt.step()
            running += loss.item()
        val_acc, val_auroc = evaluate(model, val_loader, device)
        # Select on val AUROC (robust for the class-imbalanced slide task).
        sel = val_auroc if val_auroc == val_auroc else val_acc  # nan-guard
        if sel > best_val:
            best_val = sel
            best_test = evaluate(model, test_loader, device)
            best_epoch = epoch
        log(f"epoch {epoch:02d} loss {running/len(train_loader):.4f} "
            f"val_acc {val_acc:.4f} val_auroc {val_auroc:.4f}"
            + ("  <- best" if best_epoch == epoch else ""))

    test_acc, test_auroc = best_test
    log(f"BEST @ epoch {best_epoch}: TEST acc {test_acc*100:.2f}  auroc {test_auroc*100:.2f}")

    os.makedirs(args.out, exist_ok=True)
    res = {
        "model": args.model,
        "embed_dim": dim,
        "best_epoch": best_epoch,
        "val_selection_metric": best_val,
        "test_accuracy": test_acc,
        "test_auroc": test_auroc,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }
    with open(os.path.join(args.out, f"{args.model}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"wrote {os.path.join(args.out, args.model + '.json')}")


if __name__ == "__main__":
    main()
