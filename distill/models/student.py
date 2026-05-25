from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T

from .heads import MLP

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Student(nn.Module):
    def __init__(self, teacher_dims: dict[str, int], embedding_size: int = 768,
                 backbone: nn.Module | None = None):
        super().__init__()
        # `backbone` is an injection hook for offline testing; default loads DINOv2.
        self.model = backbone if backbone is not None else torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg"
        )
        self.embedding_size = embedding_size
        self.transforms = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # One head per teacher; applied to both summary and patch tokens.
        self.mlp_heads = nn.ModuleDict(
            {name: MLP(embedding_size, dim) for name, dim in teacher_dims.items()}
        )

    def _backbone(self, batch) -> dict:
        out = self.model.forward_features(batch)
        return {"summary": out["x_norm_clstoken"], "features": out["x_norm_patchtokens"]}

    def forward(self, batch) -> dict:
        batch = self.transforms(batch)
        feats = self._backbone(batch)
        return {
            name: {
                "summary": head(feats["summary"]),
                "features": head(feats["features"]),
            }
            for name, head in self.mlp_heads.items()
        }
