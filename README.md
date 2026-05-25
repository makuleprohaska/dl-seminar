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
| `Student` | `distill/models/student.py` | DINOv2 ViT-B/14-reg backbone (86.6 M params) + one MLP projection head per teacher. Outputs `{summary, features}` projected into each teacher's embedding space. |
| `MLP` | `distill/models/heads.py` | 2-layer GELU MLP (`768 → teacher_dim`). Applied to **both** the CLS token (summary) and all 256 patch tokens (spatial features). |
| `Teacher` (ABC) | `distill/models/teachers.py` | Frozen base. Applies per-model normalisation, calls `forward_features`, then **standardises** outputs (zero-mean / unit-var per channel) (the AM-RADIO step that makes cosine loss well-scaled across teachers with different activation ranges). |
| `UNI2` | `distill/models/teachers.py` | UNI2-h via timm. `embed_dim=1536`, 8 register tokens → patches at `out[:, 9:]`. |
| `Virchow2` | `distill/models/teachers.py` | Virchow2 via timm. `embed_dim=1280`, 4 register tokens → patches at `out[:, 5:]`. |
| `HOptimus1` | `distill/models/teachers.py` | H-optimus-1 via timm. `embed_dim=1536`, 4 register tokens → patches at `out[:, 5:]`. Custom colour stats. |

---

### Data flow (one training step)

1. DataLoader yields an **un-normalised** `[B, 3, 224, 224]` float in `[0, 1]`.
2. **Student** normalises (ImageNet stats), runs `forward_features`, and projects the CLS token + 256 patch tokens through each per-teacher MLP head.
3. Each **Teacher** normalises (model-specific stats), runs `forward_features`, and standardises the output vectors.
4. Loss is computed per-teacher and summed.

Spatial alignment is guaranteed: patch size 14 at 224 px → **16 × 16 = 256 patch tokens** for the student and every teacher.

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
