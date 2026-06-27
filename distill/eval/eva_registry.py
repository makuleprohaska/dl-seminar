"""Register the distilled student as an `eva` backbone in eva's model registry.

This is the loader the prof asked for: it mirrors eva's shipped pathology
backbones (`src/eva/vision/models/networks/backbones/pathology/*.py`, e.g.
`pathology/uni2_h`, `pathology/virchow2`) so that eva's STOCK, unmodified
configs run against our student via the registry -- exactly the same code path,
split, and recipe eva uses for the published Table-1 teachers. That is what
makes the numbers comparable; our earlier custom pipelines (the `kfold_probe.py`
5-fold CV and the `ModelFromFunction`-patched `configs/eva/*_student.yaml`) did
NOT use eva's code, which is why BACH looked off and its std was inflated.

Usage on the GPU VM (eva present, Python 3.10+):

    # one-time: make eva discover this module. Either copy this file into
    #   eva/src/eva/vision/models/networks/backbones/pathology/   and add it to
    #   that package's __init__.py, OR (repo root on PYTHONPATH) add one line to
    #   that __init__.py:
    #       import distill.eval.eva_registry  # noqa: F401  -> registers the model

    export PYTHONPATH=$rootdir            # repo root, so `distill` + `eva` both import
    export MODEL_NAME=pathology/patho_distill
    export IN_FEATURES=768                # student CLS dim
    export CHECKPOINT_PATH=$rootdir/checkpoints/patho-099613.ckpt   # final student ckpt
    export DATA_ROOT=/path/to/BACH

    cd eva/src
    python3.12 -m eva predict_fit \
      --config $rootdir/eva/configs/vision/pathology/offline/classification/bach.yaml

The teachers need no loader -- they already ship in eva's registry
(`pathology/uni2_h`, `pathology/virchow2`, `pathology/h_optimus_1`); just set
`MODEL_NAME`/`IN_FEATURES` accordingly and run the same stock config.
"""

from __future__ import annotations

import os

import torch.nn as nn

from distill.eval.student_backbone import EMBED_DIM, load_student_backbone

# Registry key referenced by stock configs via `MODEL_NAME=pathology/patho_distill`.
MODEL_NAME = "pathology/patho_distill"

# Final distilled student checkpoint (handoff: checkpoints/patho-099613.ckpt).
# Overridable per run with CHECKPOINT_PATH so the same registry entry can probe
# any checkpoint without code changes.
_DEFAULT_CHECKPOINT = "checkpoints/patho-099613.ckpt"


def build_patho_distill(
    checkpoint_path: str | None = None,
    out_indices=None,  # noqa: ANN001 - accepted for eva-registry call compatibility
    **_ignored,
) -> nn.Module:
    """Factory eva's registry calls to instantiate the backbone.

    Returns the frozen student DINOv2 backbone exposing the CLS embedding
    `[B, 768]` for linear-probe classification -- the same interface eva's
    pathology backbones expose. `out_indices` and any other kwargs eva may pass
    (it threads them through for feature-pyramid models) are ignored: this is a
    pooled-CLS classification backbone, not a multi-stage feature extractor.

    `checkpoint_path` falls back to the CHECKPOINT_PATH env var, then to the
    final student checkpoint, so stock eva configs need not know the path.
    """
    ckpt = checkpoint_path or os.environ.get("CHECKPOINT_PATH") or _DEFAULT_CHECKPOINT
    return load_student_backbone(checkpoint_path=ckpt, normalize=False)


# H-optimus-1 (our actual teacher) is NOT in eva's registry -- eva ships only
# H-optimus-0 (`pathology/bioptimus_h_optimus_0`). Mirror eva's own H-optimus-0
# entry verbatim with the -1 repo so the backbone goes through the identical eva
# code path (same forward/pooling) for comparability.
H_OPTIMUS_1_NAME = "pathology/bioptimus_h_optimus_1"

# H-optimus uses model-specific normalization, NOT ImageNet. Set these as the
# eva dataset transform stats (NORMALIZE_MEAN/STD) when embedding this model.
H_OPTIMUS_1_MEAN = (0.707223, 0.578729, 0.703617)
H_OPTIMUS_1_STD = (0.211883, 0.230117, 0.177517)


def build_h_optimus_1(dynamic_img_size: bool = True, out_indices=None) -> nn.Module:
    """eva-registry factory for the Bioptimus H-optimus-1 teacher (1536-d).

    Verbatim mirror of eva's `pathology/bioptimus_h_optimus_0` entry, repo swapped
    to H-optimus-1. `dynamic_img_size` defaults to True to match eva's own
    H-optimus-0 entry exactly (a no-op at the native 224 px eval resolution, but
    kept identical so the backbone goes through eva's framework with zero
    divergence from how eva runs its other teachers).
    """
    import timm

    return timm.create_model(
        model_name="hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=dynamic_img_size,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


def register() -> bool:
    """Register our backbones in eva's registry. Returns True on success.

    Kept separate from import side-effects so this module is importable (and the
    builders testable) even where eva is absent (e.g. the local Python 3.9 env).
    Verified against eva 0.4.5: the registry is `backbone_registry` (an
    `eva.core.utils.registry.Registry`) and the decorator is
    `backbone_registry.register("pathology/<name>")`, exactly as eva's own
    pathology backbones (`pathology/mahmood_uni2_h`, `pathology/paige_virchow2`)
    register themselves.
    """
    try:
        from eva.vision.models.networks.backbones.registry import backbone_registry
    except ImportError:
        return False
    backbone_registry.register(MODEL_NAME)(build_patho_distill)
    backbone_registry.register(H_OPTIMUS_1_NAME)(build_h_optimus_1)
    return True


# Register on import so adding `import distill.eval.eva_registry` to eva's
# pathology backbone __init__.py (or copying this file there) is enough.
register()
