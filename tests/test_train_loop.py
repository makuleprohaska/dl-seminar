"""End-to-end offline run of the real LightningModel training loop.

Uses fake teachers + a fake student backbone (injection hooks) so we exercise
the actual loss aggregation, optimizer, LR schedule, and Lightning wiring
without any network / gated downloads.
"""
import pytest
import torch
import torch.nn as nn

pl = pytest.importorskip("pytorch_lightning")

from distill.data import get_dummy_dataloader
from distill.lightning_module import LightningModel
from distill.models.student import Student


class FakeBackbone(nn.Module):
    """Mimics DINOv2 forward_features: 224/14 -> 16x16 = 256 patch tokens."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=14, stride=14)

    def forward_features(self, x):
        patches = self.proj(x).flatten(2).transpose(1, 2)  # [B, 256, dim]
        return {"x_norm_clstoken": patches.mean(1), "x_norm_patchtokens": patches}


class FakeTeacher(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding_dim = dim
        self.proj = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        feats = self.proj(x).flatten(2).transpose(1, 2)  # [B, 256, dim]
        return {"summary": feats.mean(1), "features": feats}


def test_training_loop_runs_and_updates_student():
    teachers = {"t1": FakeTeacher(64), "t2": FakeTeacher(96)}
    dims = {k: v.embedding_dim for k, v in teachers.items()}
    student = Student(dims, backbone=FakeBackbone())
    model = LightningModel(teachers=teachers, student=student, max_steps=4, warmup_steps=2)

    before = student.mlp_heads["t1"][0].weight.detach().clone()

    trainer = pl.Trainer(
        max_steps=4,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, get_dummy_dataloader(batch_size=2, n=16))

    after = student.mlp_heads["t1"][0].weight.detach()
    # Teachers frozen, student trained -> head weights must move.
    assert not torch.allclose(before, after)
    # Teacher params received no gradients.
    assert all(p.grad is None for p in model.teachers.parameters())
