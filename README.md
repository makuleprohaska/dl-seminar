# patho-distill

Multi-teacher knowledge distillation for pathology foundation models. A single compact student is trained to simultaneously match three large frozen pathology experts.

---

## Architecture

### Concept

```
Unlabelled pathology tile (224×224)
          │
          ├──► Teacher: UNI2-h      (ViT-H/14, 1536-d)  ─┐
          ├──► Teacher: Virchow2    (ViT-L/14, 1280-d)  ─┼──► AM-RADIO loss
          └──► Teacher: H-optimus-1 (ViT-H/14, 1536-d)  ─┘
          │                                               ▲
          └──► Student: DINOv2 ViT-B/14-reg (768-d)
                         └── MLP heads (one per teacher)
```

All four models see the **same pixels** (geometry applied once in the DataLoader; per-model colour normalisation happens inside each wrapper). Teachers are fully frozen throughout training.

---

### Components

| Class | File | Role |
|---|---|---|
| `LightningModel` | `distill/lightning_module.py` | Training harness. Runs `_step` for train/val, aggregates per-teacher losses, owns AdamW + cosine LR schedule. |
| `Student` | `distill/models/student.py` | DINOv2 ViT-B/14-reg backbone (86.6 M params) + one MLP projection head per teacher. Outputs `{summary, features}` projected into each teacher's embeddin[...] |
| `MLP` | `distill/models/heads.py` | 2-layer GELU MLP (`768 → teacher_dim`). Applied to **both** the CLS token (summary) and all 256 patch tokens (spatial features). |
| `Teacher` (ABC) | `distill/models/teachers.py` | Frozen base. Applies per-model normalisation, calls `forward_features`, then **standardises** outputs (zero-mean / unit-var per channel) (the AM-[...] |
| `UNI2` | `distill/models/teachers.py` | UNI2-h via timm. `embed_dim=1536`, 8 register tokens → patches at `out[:, 9:]`. |
| `Virchow2` | `distill/models/teachers.py` | Virchow2 via timm. `embed_dim=1280`, 4 register tokens → patches at `out[:, 5:]`. |
| `HOptimus1` | `distill/models/teachers.py` | H-optimus-1 via timm. `embed_dim=1536`, 4 register tokens → patches at `out[:, 5:]`. Custom colour stats. |

---

### Data flow (one training step)

1. **Raw image enters:** The DataLoader provides a batch of images `[B, 3, 224, 224]` with pixel values in `[0, 1]` (unnormalised).

2. **Student processes the image:** The student model normalises the image using ImageNet statistics, extracts features, and produces:
   - 1 **summary token** (CLS token): a 768-dimensional vector for the whole image
   - 256 **patch tokens**: a 16×16 grid of spatial features (768-d each)
   - Both are then projected through MLP heads, one for each teacher, to match that teacher's embedding size.

3. **Teachers process the same image:** All three teachers receive the identical pixels. Each applies its own normalisation (UNI2, Virchow2, and H-optimus-1 use different statistics), extracts the same 1+256 token structure, and **standardises** the outputs (zero-mean, unit-variance per channel).

4. **Loss computation:** The student's projected tokens are compared to each teacher's tokens. Three losses are computed (one per teacher) and summed.

**Spatial alignment is guaranteed:** All models use patch size 14 on 224 px images, producing exactly **16 × 16 = 256 patch tokens** for both student and every teacher. Patch #i from the student aligns spatially with patch #i from each teacher.

---

### Loss

```
L_summary  = cosine_distance(student_CLS_proj,    teacher_CLS)
L_features = cosine_distance(student_patch_proj,  teacher_patches)
           + smooth_L1      (student_patch_proj,  teacher_patches)

L_total = Σ_teachers  w_summary · L_summary  +  w_feature · L_features
```

Both weights default to 1.0 (`distill/config.py`).

---

### Training configuration

| Knob | Value |
|---|---|
| Student backbone | DINOv2 ViT-B/14-reg |
| Teachers | UNI2-h · Virchow2 · H-optimus-1 |
| Batch size | 64 |
| Optimiser | AdamW, lr 1e-4, wd 0.05 |
| LR schedule | Cosine with 2 000-step linear warmup |
| Total steps | 100 000 |
| Precision | bf16-mixed (auto on GPU) |
| Grad clip | 1.0 |
| Data | 51 WebDataset shards (~248 GB JPEGs); train 0–49, val shard 50 |

---

## Benchmarking

Downstream quality is measured with **[Kaiko `eva`](https://kaiko-ai.github.io/eva/)** — the
standard linear-probe protocol behind the [pathology leaderboard](https://github.com/kaiko-ai/eva#-leaderboards),
so our numbers are directly comparable to other foundation models.

### What's evaluated

`eva` freezes the backbone, caches its embeddings, then trains a small probe head per dataset.
All four target datasets are **classification**, so eva uses the student's **CLS/summary
embedding** (768-d) — not the patch tokens:

| Dataset | Task | Head | Metric | Auto-download |
|---|---|---|---|---|
| **BACH** | 4-class patch | Linear/MLP | accuracy | yes |
| **MHIST** | binary patch | Linear/MLP | balanced accuracy | no (license form) |
| **Camelyon16** | slide-level (MIL) | ABMIL | balanced accuracy | no (large) |
| **PANDA-small** | slide-level (MIL) | ABMIL | balanced accuracy | no (large) |

> The dense **patch-token** path (`StudentBackbone.forward_patches → [B,768,16,16]`) is implemented
> too, for future segmentation tasks (MoNuSAC/CoNSeP/BCSS) — none of which are in the current set.

### How it's wired

- `distill/eval/student_backbone.py` — frozen feature extractor over the distilled DINOv2 backbone
  (drops the per-teacher distillation heads). `load_student_backbone(checkpoint_path=...)` loads the
  `student.model.*` weights from a training checkpoint. **It does not re-normalize**: eva normalizes
  with the same ImageNet stats the student trains on.
- `configs/eva/student_backbone.yaml` — backbone override; reuses eva's maintained dataset configs
  for everything else.
- `benchmark/run_eva.py` — sets the env vars (`IN_FEATURES=768`, `DATA_ROOT`, …) and runs
  `eva predict_fit` per dataset, then prints a results summary.

### Running it (on the VM)

```bash
pip install kaiko-eva
python benchmark/run_eva.py \
    --checkpoint checkpoints/last.ckpt \
    --datasets bach mhist camelyon16 panda_small \
    --data-root ./data --download        # --download only helps BACH
```

> **Not yet validated end-to-end** — eva isn't installed locally, so the CLI/config integration
> must be confirmed on the VM. The backbone wrapper and probe logic are unit-tested offline
> (`tests/test_eva_backbone.py`). Use `--dry-run` to print the exact eva commands first.

### Optional: online benchmark during training

`distill/eval/online_eval.py` provides `OnlineProbeCallback` — every *N* steps it extracts CLS
embeddings on a small labeled probe set, fits a torch linear probe, and logs accuracy
(`online/<name>_acc`) for a learning curve. Torch-only, disabled by default; see the commented
example in `train.py`.
