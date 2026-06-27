"""Streaming CAMELYON16 patch-embedder for slide-level MIL.

Disk is the blocker (full set ~700 GB, VM has ~20 GB free), so we never hold
more than one whole-slide image at a time:

    download one .tif  ->  tile foreground grid  ->  embed with each backbone
    ->  persist a per-slide patch-embedding "bag"  ->  delete the .tif

The cached bags (one [N_patches, D] tensor per slide, per model) are tiny and
are what an ABMIL head later trains on. Each slide is tiled exactly once and the
[0,1] patches are normalized once with ImageNet stats, then forwarded through
every requested backbone in the same pass -- this matches the normalization the
BACH/MHIST offline CLS probes used (eva's ResizeAndCrop default mean/std), so the
CAMELYON16 numbers stay directly comparable to the other two tasks.

Source: AWS Registry of Open Data bucket `camelyon-dataset` (us-west-2), flat
`CAMELYON16/images/<slide>.tif`; test labels in `CAMELYON16/evaluation/reference.csv`.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
import urllib.request

import torch
from torch.utils.data import DataLoader

from eva.vision.data.datasets import wsi
from eva.vision.data.transforms.common import ResizeAndCrop
from eva.vision.data.wsi.patching import samplers

from distill.eval.student_backbone import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_student_backbone,
)
from distill.eval.teacher_backbone import load_teacher_backbone

S3 = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16"

# Embedding dim per model (used only for a sanity assert on the produced bag).
MODEL_DIM = {"student": 768, "uni2": 1536, "virchow2": 1280, "hoptimus1": 1536}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Slide list + labels
# --------------------------------------------------------------------------- #
def slide_catalog() -> dict[str, int]:
    """Return {slide_id: label} for all 399 slides. 0=normal, 1=tumor.

    Train/val slide names carry the class (`normal_*`, `tumor_*`); test slides
    (`test_*`) get their label from `evaluation/reference.csv`.
    """
    names = []
    with urllib.request.urlopen(f"{S3}/checksums.md5", timeout=60) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            line = line.strip()
            if "images/" in line and line.endswith(".tif"):
                names.append(os.path.basename(line))
    names = sorted(set(names))

    ref: dict[str, str] = {}
    with urllib.request.urlopen(f"{S3}/evaluation/reference.csv", timeout=60) as r:
        for row in csv.reader(io.TextIOWrapper(r, encoding="utf-8")):
            if len(row) >= 2:
                ref[row[0].strip().replace(".tif", "")] = row[1].strip().lower()

    catalog: dict[str, int] = {}
    for fname in names:
        sid = fname.replace(".tif", "")
        if sid.startswith("normal"):
            catalog[sid] = 0
        elif sid.startswith("tumor"):
            catalog[sid] = 1
        elif sid.startswith("test"):
            label = ref.get(sid)
            if label not in ("normal", "tumor"):
                continue  # excluded / unlabeled test slide
            catalog[sid] = 0 if label == "normal" else 1
    return catalog


def download(sid: str, dest: str) -> None:
    url = f"{S3}/images/{sid}.tif"
    subprocess.run(
        ["curl", "-sSf", "--retry", "3", "--retry-delay", "5", "-o", dest, url],
        check=True,
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_models(names: list[str], checkpoint: str, device: str) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for name in names:
        log(f"loading backbone: {name}")
        if name == "student":
            m = load_student_backbone(checkpoint_path=checkpoint, normalize=False)
        else:
            m = load_teacher_backbone(teacher_name=name)  # needs HF_HUB_OFFLINE=1
        models[name] = m.to(device).eval()
    return models


# --------------------------------------------------------------------------- #
# Per-slide embedding
# --------------------------------------------------------------------------- #
def embed_slide(
    sid: str,
    tif_path: str,
    models: dict[str, torch.nn.Module],
    out_root: str,
    label: int,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> int:
    """Tile one slide once, forward through every model, save one bag per model.

    Returns the number of patches sampled.
    """
    sampler = samplers.ForegroundGridSampler(max_samples=max_samples)
    # Raw WSI regions come out at the slide's native level (~230 px, not a
    # multiple of the ViT's 14-px patch), so ResizeAndCrop is mandatory: it
    # resizes/crops to exactly 224 and applies the SAME ImageNet normalization
    # the BACH/MHIST CLS probes used -> CAMELYON16 stays comparable.
    dataset = wsi.MultiWsiDataset(
        root=os.path.dirname(tif_path),
        file_paths=[os.path.basename(tif_path)],
        width=224,
        height=224,
        sampler=sampler,
        target_mpp=0.5,
        backend="openslide",
        image_transforms=ResizeAndCrop(size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    )
    dataset.setup()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    bags: dict[str, list[torch.Tensor]] = {name: [] for name in models}

    with torch.no_grad():
        for batch in loader:
            # batch is already [B,3,224,224] float, ImageNet-normalized.
            x = batch.to(device)
            for name, model in models.items():
                bags[name].append(model(x).float().cpu())

    n_patches = 0
    for name, parts in bags.items():
        bag = torch.cat(parts, dim=0) if parts else torch.empty(0, MODEL_DIM[name])
        n_patches = bag.shape[0]
        assert bag.shape[0] == 0 or bag.shape[1] == MODEL_DIM[name], (
            f"{name}: expected dim {MODEL_DIM[name]}, got {tuple(bag.shape)}"
        )
        model_dir = os.path.join(out_root, name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save({"embeddings": bag, "label": label, "slide_id": sid},
                   os.path.join(model_dir, f"{sid}.pt"))
    return n_patches


def already_done(sid: str, names: list[str], out_root: str) -> bool:
    return all(os.path.isfile(os.path.join(out_root, n, f"{sid}.pt")) for n in names)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="student",
                    help="comma list of {student,uni2,virchow2,hoptimus1}")
    ap.add_argument("--checkpoint", default="checkpoints/patho-099613.ckpt")
    ap.add_argument("--out", default="data/camelyon16_embeddings")
    ap.add_argument("--tmp", default="data/cam_tmp")
    ap.add_argument("--max-samples", type=int, default=1000,
                    help="foreground patches per slide (fidelity vs. speed)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="embed at most N slides this run (0 = all). For calibration.")
    ap.add_argument("--keep-tif", action="store_true", help="don't delete .tif (debug)")
    args = ap.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    log(f"models={names} device={device} max_samples={args.max_samples}")
    catalog = slide_catalog()
    log(f"catalog: {len(catalog)} labeled slides "
        f"({sum(v == 0 for v in catalog.values())} normal / "
        f"{sum(v == 1 for v in catalog.values())} tumor)")

    pending = [s for s in catalog if not already_done(s, names, args.out)]
    log(f"{len(catalog) - len(pending)} already embedded, {len(pending)} pending")

    models = load_models(names, args.checkpoint, device)

    done = 0
    for sid in pending:
        if args.limit and done >= args.limit:
            break
        tif = os.path.join(args.tmp, f"{sid}.tif")
        t0 = time.time()
        try:
            log(f"-> {sid} (label={catalog[sid]}) downloading")
            download(sid, tif)
            dl = time.time() - t0
            sz = os.path.getsize(tif) / 1e9
            t1 = time.time()
            n = embed_slide(sid, tif, models, args.out, catalog[sid],
                            args.max_samples, args.batch_size, args.num_workers, device)
            emb = time.time() - t1
            log(f"   {sid}: {n} patches | dl {dl:.0f}s ({sz:.2f}GB) | embed {emb:.0f}s "
                f"| total {time.time()-t0:.0f}s")
        except Exception as e:  # noqa: BLE001  -- keep the unattended run alive
            log(f"   !! {sid} FAILED: {e}")
        finally:
            if not args.keep_tif and os.path.isfile(tif):
                os.remove(tif)
        done += 1

    log(f"RUN END: embedded {done} slides this invocation")


if __name__ == "__main__":
    main()
