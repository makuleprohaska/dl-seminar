from __future__ import annotations

import os
import re

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# Nextcloud public share holding the WebDataset shards (shard_0_0.tar .. shard_0_50.tar).
DATA_BASE_URL = os.environ.get(
    "DATA_BASE_URL", "https://hmgubox2.helmholtz-muenchen.de/public.php/webdav"
)
DATA_SHARE_TOKEN = os.environ.get("DATA_SHARE_TOKEN")  # the public-share token
NUM_SHARDS = 51
IMAGE_SIZE = 224

# Held-out split: train on shards 0..49, validate on shard 50 (inclusive ranges).
TRAIN_SHARDS = (0, 49)
VAL_SHARDS = (50, 50)


def make_transform(train: bool = True):
    """Tiles are 256x256. Emit an UN-normalized [3,224,224] float in [0,1].

    The 256->224 geometric step is applied once here so the student and all
    teachers see identical pixels (spatially-aligned features). Per-model
    normalization happens inside each model wrapper, not here.
    """
    if train:
        return T.Compose([T.RandomCrop(IMAGE_SIZE), T.RandomHorizontalFlip(), T.ToTensor()])
    return T.Compose([T.CenterCrop(IMAGE_SIZE), T.ToTensor()])


class DecodeTransform:
    """Pick the image field from a WebDataset sample and apply the transform.

    A module-level class rather than a closure/lambda so it survives the
    `spawn` DataLoader start method (macOS/Windows) — a lambda is unpicklable
    and crashes `num_workers>0` there.

    Accepts any common image extension (the in-tar extension of the real shards
    isn't guaranteed to be `.jpg`); broad coverage makes a systemic key mismatch
    that would silently stream an empty dataset effectively impossible for real
    JPEG shards. A genuinely missing image field raises a descriptive KeyError.
    """

    IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")

    def __init__(self, train: bool = True):
        self.transform = make_transform(train)

    def __call__(self, sample):
        for key in self.IMAGE_KEYS:
            if key in sample:
                return self.transform(sample[key])
        raise KeyError(
            f"no image field in sample (keys={sorted(sample)}); "
            f"expected one of {self.IMAGE_KEYS}"
        )


# --------------------------------------------------------------------------- #
# Placeholder loader (no network) — kept for offline smoke runs.
# --------------------------------------------------------------------------- #
class DummyDataset(Dataset):
    def __init__(self, n: int = 64, size: int = IMAGE_SIZE):
        self.n, self.size = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.ones(3, self.size, self.size)


def get_dummy_dataloader(batch_size: int = 2, n: int = 64, size: int = IMAGE_SIZE):
    return DataLoader(DummyDataset(n, size), batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Real streaming loader — WebDataset reads the tar shards directly off the
# Nextcloud WebDAV endpoint via curl (no local 248 GB download required).
# --------------------------------------------------------------------------- #
def _shard_url(shard_range: tuple[int, int]) -> str:
    if not DATA_SHARE_TOKEN:
        raise RuntimeError("Set DATA_SHARE_TOKEN to the Nextcloud public-share token.")
    start, end = shard_range
    return (
        f"pipe:curl -s -f -u {DATA_SHARE_TOKEN}: "
        f"{DATA_BASE_URL}/shard_0_{{{start}..{end}}}.tar"
    )


def _num_shards_in_url(url: str) -> int | None:
    """Shard count from a brace range like `shard_0_{0..49}.tar`; None if unknown."""
    m = re.search(r"\{(\d+)\.\.(\d+)\}", url)
    return int(m.group(2)) - int(m.group(1)) + 1 if m else None


def get_webdataset_loader(batch_size: int = 64, num_workers: int = 4,
                          train: bool = True, shuffle: int = 1000,
                          url: str | None = None,
                          shard_range: tuple[int, int] | None = None):
    import webdataset as wds

    if url is None:
        url = _shard_url(shard_range or (TRAIN_SHARDS if train else VAL_SHARDS))
    # WebDataset's split_by_worker hands each worker a disjoint subset of shards;
    # with more workers than shards the extra workers get nothing and empty_check
    # aborts the run (the val split is a single shard). Cap workers to shard count.
    n_shards = _num_shards_in_url(url)
    if n_shards is not None:
        num_workers = min(num_workers, n_shards)
    dataset = (
        wds.WebDataset(
            url,
            shardshuffle=train,
            handler=wds.warn_and_continue,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(shuffle if train else 0)
        .decode("pil", handler=wds.warn_and_continue)
        .map(DecodeTransform(train), handler=wds.warn_and_continue)
        .batched(batch_size, collation_fn=torch.utils.data.default_collate)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)


def get_dataloader(batch_size: int = 2, **kwargs):
    """Real streaming loader when DATA_SHARE_TOKEN is set, else the dummy loader."""
    if DATA_SHARE_TOKEN:
        return get_webdataset_loader(batch_size=batch_size, **kwargs)
    print("WARNING: DATA_SHARE_TOKEN not set; falling back to DummyDataset (torch.ones).")
    return get_dummy_dataloader(batch_size=batch_size)
