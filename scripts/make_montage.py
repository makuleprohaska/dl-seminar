"""Class-stratified tile montages of the benchmark datasets, for the presentation.

PIL-only (no matplotlib). One row per class, K example tiles per row, with a class
label strip on the left and a title. Produces a clean PNG per dataset.

    # on the VM (raw images live there):
    python scripts/make_montage.py bach  --root data/bach/ICIAR2018_BACH_Challenge/Photos --out presentation/bach_montage.png
    python scripts/make_montage.py mhist --root data/mhist --out presentation/mhist_montage.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int):
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def bach_rows(root: str, k: int, seed: int):
    """(class, [tiles]) for BACH: subdirs Benign/InSitu/Invasive/Normal."""
    rng = random.Random(seed)
    rows = []
    for cls in ["Benign", "InSitu", "Invasive", "Normal"]:
        files = sorted(glob.glob(os.path.join(root, cls, "*.tif")))
        picks = rng.sample(files, min(k, len(files)))
        rows.append((cls, [Image.open(f).convert("RGB") for f in picks]))
    return rows


def mhist_rows(root: str, k: int, seed: int):
    """(class, [tiles]) for MHIST: annotations.csv maps name -> SSA/HP (test split)."""
    rng = random.Random(seed)
    by_cls: dict[str, list[str]] = {"HP": [], "SSA": []}
    with open(os.path.join(root, "annotations.csv")) as f:
        for r in csv.DictReader(f):
            if r["Partition"] == "test":  # show what we evaluate on
                by_cls[r["Majority Vote Label"]].append(r["Image Name"])
    label = {"HP": "HP (benign)", "SSA": "SSA (precursor)"}
    rows = []
    for cls in ["HP", "SSA"]:
        picks = rng.sample(by_cls[cls], min(k, len(by_cls[cls])))
        imgs = [Image.open(os.path.join(root, "images", n)).convert("RGB") for n in picks]
        rows.append((label[cls], imgs))
    return rows


def montage(rows, thumb, title, out, pad=8):
    tw, th = thumb
    ncol = max(len(imgs) for _, imgs in rows)
    cell_w, cell_h = tw + pad, th + pad
    title_h = 52
    # Auto-size the left label strip to the widest class label so nothing clips.
    lf = _font(20)
    _m = Image.new("RGB", (1, 1))
    _d = ImageDraw.Draw(_m)
    label_w = pad + max(int(_d.textlength(label, font=lf)) for label, _ in rows) + pad
    W = label_w + ncol * cell_w + pad
    H = title_h + len(rows) * cell_h + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad + 4), title, fill=(20, 20, 20), font=_font(30))
    for r, (label, imgs) in enumerate(rows):
        y = title_h + r * cell_h
        draw.text((pad, y + th // 2 - 12), label, fill=(20, 20, 20), font=_font(20))
        for c, im in enumerate(imgs):
            canvas.paste(im.resize(thumb, Image.LANCZOS), (label_w + c * cell_w, y))
    canvas.save(out, dpi=(220, 220))
    print(f"wrote {out}  ({W}x{H}, {len(rows)} classes x {ncol} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bach", "mhist"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=6, help="tiles per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.dataset == "bach":
        rows = bach_rows(args.root, args.k, args.seed)
        montage(rows, (256, 192), "BACH - breast histology (4 classes, 2048x1536 H&E)", args.out)
    else:
        rows = mhist_rows(args.root, args.k, args.seed)
        montage(rows, (192, 192), "MHIST - colorectal polyps (test split, 224x224 H&E)", args.out)


if __name__ == "__main__":
    main()
