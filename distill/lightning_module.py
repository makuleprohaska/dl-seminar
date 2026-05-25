from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from . import losses
from .models.student import Student
from .models.teachers import TEACHERS


class LightningModel(pl.LightningModule):
    def __init__(self, teacher_names: list[str] | None = None, lr: float = 1e-4,
                 weight_decay: float = 0.05, w_summary: float = 1.0,
                 w_feature: float = 1.0, max_steps: int = 100_000,
                 warmup_steps: int = 2_000, teachers: dict | None = None,
                 student: nn.Module | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher_names", "teachers", "student"])
        # `teachers`/`student` are injection hooks for offline testing.
        if teachers is None:
            teacher_names = teacher_names or list(TEACHERS)
            teachers = {n: TEACHERS[n]() for n in teacher_names}
        self.teachers = nn.ModuleDict(teachers)
        teacher_dims = {n: t.embedding_dim for n, t in self.teachers.items()}
        self.student = student if student is not None else Student(teacher_dims)
        self.lr = lr
        self.weight_decay = weight_decay
        self.w_summary = w_summary
        self.w_feature = w_feature
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

    def train(self, mode: bool = True):
        # Keep teachers in eval regardless of train/val phase.
        super().train(mode)
        for t in self.teachers.values():
            t.eval()
        return self

    def _step(self, batch, stage: str):
        student_out = self.student(batch)
        total = 0.0
        for name, teacher in self.teachers.items():
            with torch.no_grad():
                t = teacher(batch)
            s = student_out[name]
            # Guard the spatial alignment we rely on (student 256 == teacher 256).
            assert s["features"].shape[1] == t["features"].shape[1], (
                f"Patch-token mismatch for {name}: student {s['features'].shape[1]} "
                f"vs teacher {t['features'].shape[1]}. Spatial loss needs equal counts."
            )
            l_sum = losses.summary_loss(s["summary"], t["summary"])
            l_feat = losses.feature_loss(s["features"], t["features"])
            self.log(f"{stage}/{name}_summary", l_sum)
            self.log(f"{stage}/{name}_feature", l_feat)
            total = total + self.w_summary * l_sum + self.w_feature * l_feat
        self.log(f"{stage}/loss", total, prog_bar=True)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Only the student backbone + projection heads are trained.
        opt = torch.optim.AdamW(
            self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
