"""Offline tests for the online-probe helpers (torch-only, no Lightning)."""
import torch

from distill.eval.online_eval import balanced_accuracy, fit_linear_probe


def test_balanced_accuracy_perfect_and_chance():
    y = torch.tensor([0, 0, 1, 1])
    assert balanced_accuracy(y, y, num_classes=2) == 1.0
    # All-wrong predictions -> 0 recall on both classes.
    assert balanced_accuracy(1 - y, y, num_classes=2) == 0.0


def test_balanced_accuracy_handles_class_imbalance():
    # 3 of class 0 (all right), 1 of class 1 (wrong): plain acc .75, balanced .5
    targets = torch.tensor([0, 0, 0, 1])
    preds = torch.tensor([0, 0, 0, 0])
    assert balanced_accuracy(preds, targets, num_classes=2) == 0.5


def test_fit_linear_probe_separates_easy_data():
    g = torch.Generator().manual_seed(0)
    # Two well-separated Gaussian blobs in 32-d.
    def blob(center, n):
        return torch.randn(n, 32, generator=g) * 0.3 + center

    c0, c1 = torch.zeros(32), torch.ones(32) * 3
    x_tr = torch.cat([blob(c0, 64), blob(c1, 64)])
    y_tr = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    x_va = torch.cat([blob(c0, 32), blob(c1, 32)])
    y_va = torch.cat([torch.zeros(32), torch.ones(32)]).long()

    m = fit_linear_probe(x_tr, y_tr, x_va, y_va, num_classes=2, epochs=200)
    assert m["acc"] > 0.95
    assert m["balanced_acc"] > 0.95
