"""Offline wiring tests (no network / no gated downloads).

Full integration with the real teachers is exercised via `python train.py`
once HF_TOKEN is set.
"""
import torch
from PIL import Image

from distill import losses
from distill.data import make_transform
from distill.models.heads import MLP


def test_shard_url_brace_range(monkeypatch):
    import distill.data as data

    monkeypatch.setattr(data, "DATA_SHARE_TOKEN", "TESTTOKEN")
    url = data._shard_url((0, 49))
    assert "shard_0_{0..49}.tar" in url
    assert "TESTTOKEN" in url


def test_transform_crops_256_to_224_unnormalized():
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    out = make_transform(train=True)(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0  # not normalized


def test_mlp_projects_summary_and_patches():
    head = MLP(in_dim=768, out_dim=1536)
    assert head(torch.randn(2, 768)).shape == (2, 1536)
    assert head(torch.randn(2, 256, 768)).shape == (2, 256, 1536)


def test_losses_are_finite_scalars():
    s_sum, t_sum = torch.randn(2, 1536), torch.randn(2, 1536)
    s_feat, t_feat = torch.randn(2, 256, 1536), torch.randn(2, 256, 1536)
    assert torch.isfinite(losses.summary_loss(s_sum, t_sum))
    assert torch.isfinite(losses.feature_loss(s_feat, t_feat))


def test_perfect_match_has_zero_cosine_distance():
    x = torch.randn(2, 256, 1536)
    assert torch.isclose(losses.cosine_distance(x, x), torch.tensor(0.0), atol=1e-5)
