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

Downstream quality is measured with **[Kaiko `eva`](https://kaiko-ai.github.io/eva/) 0.4.5**: every
model — the student **and** all three teachers — runs through eva's stock offline-classification
configs end-to-end via `eva predict_fit` (eva's linear-probe protocol, `n_runs=5`), so the numbers
are directly comparable across models and to published leaderboards.

### Results (eva stock config, `n_runs=5`)

| Model | Params | BACH (mc acc) | MHIST (bal acc) | CAMELYON16 |
|---|---|---|---|---|
| UNI2-h | 681 M | **91.46 ± 0.29** | 82.27 ± 0.11 | 88.37 |
| Virchow2 | 632 M | 88.00 ± 0.76 | **86.11 ± 0.05** | **96.12** |
| H-optimus-1 | 1.1 B | 78.07 ± 0.70 | 83.79 ± 0.13 | 92.25 |
| **Student (ours)** | 86.6 M | 85.16 ± 0.54 | 84.39 ± 0.04 | 93.80 |

BACH and MHIST are patch-level linear probes (multiclass accuracy / binary balanced accuracy,
averaged over 5 runs); CAMELYON16 is a single-fold slide-level ABMIL run. The student stays
competitive across all three tasks — second on MHIST and CAMELYON16 — and surpasses the
1.1 B-param H-optimus-1 on BACH and MHIST at 7–13× fewer parameters. Best per column in **bold**.

### What's evaluated

`eva` freezes the backbone, embeds each dataset, then trains a small probe head:

| Dataset | Task | Head | Metric |
|---|---|---|---|
| **BACH** | 4-class patch | Linear | multiclass accuracy |
| **MHIST** | binary patch | Linear | binary balanced accuracy |
| **CAMELYON16** | slide-level (MIL) | ABMIL | accuracy / AUROC |

Classification uses the student's **CLS/summary** embedding (768-d). The dense **patch-token** path
(`StudentBackbone.forward_patches → [B,768,16,16]`) is implemented for future segmentation work.

### How it's wired

- **`distill/eval/eva_registry.py`** — registers the distilled student as `pathology/patho_distill`
  in eva's backbone registry (plus a `pathology/bioptimus_h_optimus_1` entry, since eva ships
  H-optimus-0), so eva's stock configs run against our model unchanged — the same registry mechanism
  eva's own pathology backbones use (`backbone_registry.register("pathology/<name>")`).
- **`distill/eval/student_backbone.py`** — frozen feature extractor over the distilled DINOv2
  backbone (drops the per-teacher distillation heads); exposes the 768-d CLS embedding.
- To let a pip-installed eva discover the loader, add `import distill.eval.eva_registry` to its
  `…/backbones/pathology/__init__.py` and run with `PYTHONPATH=<repo root>`.

### Running it (on the GPU VM)

```bash
nohup bash scripts/run_eva_native.sh > logs/eva_native/all.log 2>&1 &
```

Per-model env: `MODEL_NAME=pathology/<name>`, `IN_FEATURES` (768 student / 1536 uni2 / 1280 virchow2
/ 1536 hoptimus1). The gated teachers (UNI2-h, Virchow2) require `HF_TOKEN`; H-optimus uses its own
image normalization (`NORMALIZE_MEAN`/`NORMALIZE_STD`, see `scripts/run_eva_hoptimus.sh`). The
one-page method + results writeup is `presentation/handout.tex`.

### Optional: online benchmark during training

`distill/eval/online_eval.py` provides `OnlineProbeCallback` — every *N* steps it extracts CLS
embeddings on a small labeled probe set, fits a torch linear probe, and logs accuracy
(`online/<name>_acc`) for a learning curve. Torch-only, disabled by default; see the commented
example in `train.py`.
