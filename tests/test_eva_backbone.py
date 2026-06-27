"""Offline tests for the eva-facing student backbone wrapper.

No network / GPU / gated downloads: a fake DINOv2-style backbone is injected,
mirroring tests/test_train_loop.py.
"""
import torch
import torch.nn as nn

from distill.eval.student_backbone import (
    EMBED_DIM,
    GRID,
    StudentBackbone,
    load_student_backbone,
)


class FakeBackbone(nn.Module):
    """Mimics DINOv2 forward_features: 224/14 -> 16x16 = 256 patch tokens."""

    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=14, stride=14)

    def forward_features(self, x):
        patches = self.proj(x).flatten(2).transpose(1, 2)  # [B, 256, dim]
        return {"x_norm_clstoken": patches.mean(1), "x_norm_patchtokens": patches}


def test_forward_returns_cls_embedding():
    bb = StudentBackbone(backbone=FakeBackbone())
    out = bb(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, EMBED_DIM)


def test_forward_patches_returns_spatial_feature_map():
    bb = StudentBackbone(backbone=FakeBackbone())
    out = bb.forward_patches(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, EMBED_DIM, GRID, GRID)


def test_no_double_normalization_by_default():
    # normalize=False must leave pixels untouched before the backbone.
    seen = {}

    class Spy(FakeBackbone):
        def forward_features(self, x):
            seen["x"] = x
            return super().forward_features(x)

    x = torch.rand(1, 3, 224, 224)
    StudentBackbone(backbone=Spy(), normalize=False)(x)
    assert torch.equal(seen["x"], x)  # unchanged

    StudentBackbone(backbone=Spy(), normalize=True)(x)
    assert not torch.equal(seen["x"], x)  # ImageNet-normalized


def test_backbone_is_frozen_and_eval():
    bb = StudentBackbone(backbone=FakeBackbone())
    assert not bb.training
    assert all(not p.requires_grad for p in bb.parameters())


def test_load_student_backbone_round_trip(tmp_path):
    # Build a checkpoint shaped like a LightningModel state_dict and confirm
    # only student.model.* weights are loaded into the wrapper.
    src = FakeBackbone()
    state = {f"student.model.{k}": v for k, v in src.state_dict().items()}
    state["student.mlp_heads.t1.0.weight"] = torch.randn(64, EMBED_DIM)  # ignored
    state["teachers.t1.proj.weight"] = torch.randn(64, 3, 14, 14)  # ignored
    ckpt = tmp_path / "patho.ckpt"
    torch.save({"state_dict": state}, ckpt)

    loaded = load_student_backbone(checkpoint_path=str(ckpt), backbone=FakeBackbone())
    x = torch.rand(2, 3, 224, 224)
    assert torch.allclose(loaded(x), src.forward_features(x)["x_norm_clstoken"], atol=1e-5)


def test_load_raises_when_no_backbone_weights(tmp_path):
    ckpt = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"teachers.t1.w": torch.zeros(1)}}, ckpt)
    try:
        load_student_backbone(checkpoint_path=str(ckpt), backbone=FakeBackbone())
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_eva_registry_builder_returns_cls_backbone(tmp_path, monkeypatch):
    # The eva-registry factory must load a checkpoint via CHECKPOINT_PATH and
    # return the same [B, 768] CLS interface eva's pathology backbones expose.
    # No eva / GPU needed: inject a fake DINOv2 and a fake checkpoint.
    from distill.eval import eva_registry
    import distill.eval.student_backbone as sb

    src = FakeBackbone()
    state = {f"student.model.{k}": v for k, v in src.state_dict().items()}
    ckpt = tmp_path / "patho.ckpt"
    torch.save({"state_dict": state}, ckpt)
    monkeypatch.setenv("CHECKPOINT_PATH", str(ckpt))

    # Patch the loader so we don't reach torch.hub for the real DINOv2 weights.
    real_load = sb.load_student_backbone
    monkeypatch.setattr(
        eva_registry,
        "load_student_backbone",
        lambda checkpoint_path, normalize=False: real_load(
            checkpoint_path=checkpoint_path, backbone=FakeBackbone(), normalize=normalize
        ),
    )

    # eva may thread out_indices / extra kwargs through; they must be ignored.
    model = eva_registry.build_patho_distill(out_indices=(0,), unused="x")
    out = model(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, EMBED_DIM)
