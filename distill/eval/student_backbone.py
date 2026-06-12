from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Student backbone geometry: ViT-B/14 on 224 px -> 16x16 = 256 patch tokens, 768-d.
EMBED_DIM = 768
GRID = 224 // 14  # 16


class StudentBackbone(nn.Module):
    """Frozen feature extractor for downstream evaluation (e.g. Kaiko `eva`).

    Exposes only the DINOv2 backbone of the distilled student — NOT the
    per-teacher MLP heads, which are distillation-only artifacts. Returns the
    CLS/summary embedding `[B, 768]` for classification tasks; patch tokens are
    available via `forward_patches` for future dense/segmentation tasks.

    Normalization: `eva` (and most eval harnesses) normalize pixels in their own
    dataloader, by default with the SAME ImageNet stats the student trains on.
    So `normalize=False` by default to avoid double-normalization. Set
    `normalize=True` only when feeding raw `[0, 1]` pixels directly.
    """

    def __init__(self, backbone: nn.Module | None = None, normalize: bool = False):
        super().__init__()
        # `backbone` is an injection hook for offline testing; default loads DINOv2.
        self.model = backbone if backbone is not None else torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg"
        )
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else None
        self.requires_grad_(False)
        self.eval()

    def _features(self, x) -> dict:
        if self.norm is not None:
            x = self.norm(x)
        out = self.model.forward_features(x)
        return {"summary": out["x_norm_clstoken"], "features": out["x_norm_patchtokens"]}

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        """Classification interface: CLS embedding `[B, 768]`."""
        return self._features(x)["summary"]

    @torch.no_grad()
    def forward_patches(self, x) -> torch.Tensor:
        """Dense interface: patch tokens as a feature map `[B, 768, 16, 16]`.

        eva segmentation decoders expect spatial feature maps, not the flat
        `[B, N, D]` token sequence — reshape the 256 tokens into the 16x16 grid.
        """
        tokens = self._features(x)["features"]  # [B, 256, 768]
        b, n, d = tokens.shape
        side = int(n ** 0.5)
        return tokens.transpose(1, 2).reshape(b, d, side, side)


def load_student_backbone(
    checkpoint_path: str | None = None,
    backbone: nn.Module | None = None,
    normalize: bool = False,
    map_location: str = "cpu",
) -> StudentBackbone:
    """Build a `StudentBackbone` and load the student weights from a checkpoint.

    `checkpoint_path` points at a `LightningModel` checkpoint (its `state_dict`
    holds `student.model.*`, `student.mlp_heads.*`, and `teachers.*`). Only the
    backbone (`student.model.*`) is needed for embedding extraction; the heads
    and teachers are dropped. This is the factory `eva`'s `ModelFromFunction`
    wrapper points at.
    """
    model = StudentBackbone(backbone=backbone, normalize=normalize)
    if checkpoint_path is None:
        return model

    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)
    prefix = "student.model."
    backbone_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not backbone_state:
        raise KeyError(
            f"no '{prefix}*' weights in checkpoint {checkpoint_path}; "
            f"found prefixes {sorted({k.split('.')[0] for k in state})}"
        )
    missing, unexpected = model.model.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        print(
            f"load_student_backbone: loaded {len(backbone_state)} tensors "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
    return model
