"""Optional online benchmark: a lightweight linear probe run during training.

This is the OPTIONAL extension. The primary benchmark is the offline Kaiko
`eva` run (see `benchmark/run_eva.py`); this callback gives a cheap, in-process
signal of downstream quality *during* training (e.g. every 1000 steps) so you
can plot a learning curve and compare against other models early.

It mirrors eva's offline protocol in miniature: freeze the student, extract CLS
embeddings on a small labeled probe set, fit a linear probe, log accuracy.
Deliberately torch-only (no sklearn) to keep dependencies minimal.

Not wired into `train.py` by default — see the commented example there.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Mean per-class recall — the metric eva reports for MHIST/Camelyon16/PANDA."""
    recalls = []
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            recalls.append((preds[mask] == c).float().mean())
    return float(torch.stack(recalls).mean()) if recalls else 0.0


def fit_linear_probe(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    num_classes: int | None = None, epochs: int = 100, lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    """Train a single Linear layer on frozen features; report val metrics.

    Pure function (no Lightning) so it's unit-testable offline. Features are
    standardized with train statistics before fitting.
    """
    num_classes = num_classes or int(max(y_train.max(), y_val.max())) + 1
    mean, std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True) + 1e-6
    x_train, x_val = (x_train - mean) / std, (x_val - mean) / std

    probe = torch.nn.Linear(x_train.shape[1], num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(x_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = probe(x_val).argmax(1)
    acc = float((preds == y_val).float().mean())
    return {
        "acc": acc,
        "balanced_acc": balanced_accuracy(preds, y_val, num_classes),
        "probe_train_loss": float(loss.detach()),
    }


try:
    import pytorch_lightning as pl

    class OnlineProbeCallback(pl.Callback):
        """Run a linear-probe benchmark every `every_n_steps` training steps.

        `train_loader`/`val_loader` must yield `(image[B,3,224,224] in [0,1], label[B])`
        batches for a small labeled probe set (e.g. a BACH subset). Keep them
        small — this runs inline with training.
        """

        def __init__(self, train_loader, val_loader, every_n_steps: int = 1000,
                     num_classes: int | None = None, max_batches: int | None = None,
                     name: str = "probe", probe_epochs: int = 100):
            super().__init__()
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.every_n_steps = every_n_steps
            self.num_classes = num_classes
            self.max_batches = max_batches
            self.name = name
            self.probe_epochs = probe_epochs

        @torch.no_grad()
        def _embed(self, pl_module, loader) -> tuple[torch.Tensor, torch.Tensor]:
            student = pl_module.student
            embs, labels = [], []
            for i, (x, y) in enumerate(loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                x = student.transforms(x.to(pl_module.device))  # same norm as training
                cls = student.model.forward_features(x)["x_norm_clstoken"]
                embs.append(cls.float().cpu())
                labels.append(torch.as_tensor(y))
            return torch.cat(embs), torch.cat(labels)

        def on_train_batch_end(self, trainer, pl_module, *args):
            step = trainer.global_step
            if step == 0 or step % self.every_n_steps != 0:
                return
            was_training = pl_module.training
            pl_module.eval()
            try:
                x_tr, y_tr = self._embed(pl_module, self.train_loader)
                x_va, y_va = self._embed(pl_module, self.val_loader)
                metrics = fit_linear_probe(
                    x_tr, y_tr, x_va, y_va,
                    num_classes=self.num_classes, epochs=self.probe_epochs,
                )
            finally:
                if was_training:
                    pl_module.train()
            for k, v in metrics.items():
                pl_module.log(f"online/{self.name}_{k}", v)

except ImportError:  # pragma: no cover - pl is present in the training env
    OnlineProbeCallback = None
