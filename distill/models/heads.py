from __future__ import annotations

import torch.nn as nn


class MLP(nn.Sequential):
    """Per-teacher projection head: student dim -> teacher dim."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        hidden_dim = hidden_dim or out_dim
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
