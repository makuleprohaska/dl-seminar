from __future__ import annotations

import torch
import torch.nn as nn

from distill.models.teachers import TEACHERS, IMAGENET_MEAN, IMAGENET_STD


class TeacherBackbone(nn.Module):
    """Frozen teacher feature extractor for downstream evaluation (Kaiko `eva`).

    Wraps a teacher from `distill.models.teachers` and exposes the classification
    interface eva expects: the raw CLS/summary embedding `[B, D]`.

    Two deliberate differences from the teacher's training-time `forward`:

    * **No per-batch standardization.** `Teacher.forward` standardizes outputs
      (zero-mean/unit-var per channel over the batch) for the AM-RADIO loss; that
      is batch-dependent and would corrupt embeddings extracted for a linear probe.
      We call `forward_features` and return the raw summary token instead.
    * **Correct per-teacher normalization.** The eva config normalizes pixels with
      ImageNet stats (the default, same pipeline as the student backbone). UNI2-h
      and Virchow2 expect exactly that, so their input passes straight through.
      H-optimus-1 expects custom color stats (`teachers.py`), so we invert the
      ImageNet normalization and re-apply the teacher's own — keeping a single eva
      pipeline for all teachers (no env list-parsing to get wrong).
    """

    def __init__(self, teacher_name: str):
        super().__init__()
        if teacher_name not in TEACHERS:
            raise KeyError(
                f"unknown teacher {teacher_name!r}; available: {list(TEACHERS)}"
            )
        self.teacher = TEACHERS[teacher_name]()
        self.embedding_dim = self.teacher.embedding_dim

        norm = self.teacher.transforms  # T.Normalize with this teacher's stats
        tmean = torch.as_tensor(norm.mean, dtype=torch.float32).view(1, 3, 1, 1)
        tstd = torch.as_tensor(norm.std, dtype=torch.float32).view(1, 3, 1, 1)
        imean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        istd = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # Re-normalize only when the teacher's stats differ from ImageNet.
        self._renorm = not (torch.allclose(tmean, imean) and torch.allclose(tstd, istd))
        self.register_buffer("imean", imean)
        self.register_buffer("istd", istd)
        self.register_buffer("tmean", tmean)
        self.register_buffer("tstd", tstd)

        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        if self._renorm:
            x = x * self.istd + self.imean       # undo eva's ImageNet normalization -> [0,1]
            x = (x - self.tmean) / self.tstd     # apply this teacher's own normalization
        return self.teacher.forward_features(x)["summary"]


def load_teacher_backbone(teacher_name: str, **_) -> TeacherBackbone:
    """Factory for eva's `ModelFromFunction` wrapper. `teacher_name` is one of
    the keys of `distill.models.teachers.TEACHERS` (`uni2`, `virchow2`, `hoptimus1`)."""
    return TeacherBackbone(teacher_name)
