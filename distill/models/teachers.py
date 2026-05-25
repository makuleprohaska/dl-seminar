from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.transforms as T

# NOTE: `timm` is imported lazily inside each concrete teacher's __init__ so that
# the dummy/offline path (and tests with fake teachers) need not install timm.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Teacher(nn.Module, ABC):
    """Frozen teacher. Concrete subclasses only implement forward_features()."""

    def __init__(self, model: nn.Module, embedding_dim: int,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.transforms = T.Normalize(mean=mean, std=std)
        self.requires_grad_(False)
        self.eval()

    @abstractmethod
    def forward_features(self, batch) -> dict:
        """Return {'summary': [B, D], 'features': [B, N, D]}."""

    def standardize(self, x, eps: float = 1e-6):
        # Per-channel zero-mean/unit-var over all non-channel dims (AM-RADIO).
        dims = tuple(range(x.dim() - 1))
        return (x - x.mean(dims, keepdim=True)) / (x.std(dims, keepdim=True) + eps)

    @torch.no_grad()
    def forward(self, batch) -> dict:
        batch = self.transforms(batch)
        out = self.forward_features(batch)
        return {
            "summary": self.standardize(out["summary"]),
            "features": self.standardize(out["features"]),
        }


class UNI2(Teacher):
    def __init__(self):
        import timm
        from timm.layers import SwiGLUPacked  # noqa: F401  (used via kwargs)

        kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2 * 2.66667, num_classes=0, no_embed_class=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
            dynamic_img_size=True,
        )
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs)
        super().__init__(model, embedding_dim=1536)

    def forward_features(self, batch):
        out = self.model.forward_features(batch)
        return {"summary": out[:, 0], "features": out[:, 9:]}  # 1 CLS + 8 registers


class Virchow2(Teacher):
    def __init__(self):
        import timm
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        )
        super().__init__(model, embedding_dim=1280)

    def forward_features(self, batch):
        out = self.model(batch)
        return {"summary": out[:, 0], "features": out[:, 5:]}  # 1 CLS + 4 registers


class HOptimus1(Teacher):
    MEAN = (0.707223, 0.578729, 0.703617)
    STD = (0.211883, 0.230117, 0.177517)

    def __init__(self):
        import timm

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True,
            init_values=1e-5, dynamic_img_size=False,
        )
        super().__init__(model, embedding_dim=1536, mean=self.MEAN, std=self.STD)

    def forward_features(self, batch):
        out = self.model.forward_features(batch)
        return {"summary": out[:, 0], "features": out[:, 5:]}  # 1 CLS + 4 registers


TEACHERS = {"uni2": UNI2, "virchow2": Virchow2, "hoptimus1": HOptimus1}
