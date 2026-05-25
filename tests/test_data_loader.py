"""Offline tests for the real WebDataset streaming loader.

These build tiny local tar shards (no network / no token) and run them through
the actual loader pipeline, covering the decode/transform contract, image-key
robustness, and worker capping that the real shards depend on.
"""
import io
import pickle
import tarfile

import pytest
import torch
from PIL import Image

wds = pytest.importorskip("webdataset")

from distill import data as D


def _make_shards(tmp_path, counts, ext="jpg"):
    for si, n in enumerate(counts):
        with tarfile.open(tmp_path / f"shard_0_{si}.tar", "w") as tar:
            for i in range(n):
                img = Image.new("RGB", (256, 256), (si * 40, i * 9, 100))
                buf = io.BytesIO()
                img.save(buf, "JPEG")
                blob = buf.getvalue()
                info = tarfile.TarInfo(name=f"{si:02d}{i:03d}.{ext}")
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))


def _drain(loader):
    n, shapes = 0, []
    for b in loader:
        n += b.shape[0]
        shapes.append(tuple(b.shape[1:]))
    return n, shapes


def test_decode_transform_is_picklable():
    # Guards the spawn DataLoader start method: a lambda here crashes num_workers>0.
    obj = pickle.loads(pickle.dumps(D.DecodeTransform(train=True)))
    out = obj({"jpg": Image.new("RGB", (256, 256), (10, 20, 30))})
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0  # un-normalized


def test_decode_transform_accepts_alternate_image_keys():
    img = Image.new("RGB", (256, 256), (1, 2, 3))
    for key in ("jpeg", "png", "webp"):
        assert D.DecodeTransform(train=False)({key: img}).shape == (3, 224, 224)


def test_decode_transform_raises_on_missing_image():
    # A systemic key mismatch must fail loud, not silently yield an empty dataset.
    with pytest.raises(KeyError):
        D.DecodeTransform()({"txt": "not an image"})


def test_num_shards_in_url():
    assert D._num_shards_in_url("shard_0_{0..49}.tar") == 50
    assert D._num_shards_in_url("shard_0_{50..50}.tar") == 1
    assert D._num_shards_in_url("/data/onefile.tar") is None


def test_loader_reads_every_local_jpg_sample(tmp_path):
    _make_shards(tmp_path, [10, 10, 7])
    url = f"{tmp_path}/shard_0_{{0..2}}.tar"
    n, shapes = _drain(D.get_webdataset_loader(4, num_workers=0, train=True, shuffle=5, url=url))
    assert n == 27  # no silent drops
    assert all(s == (3, 224, 224) for s in shapes)


def test_loader_decodes_jpeg_extension(tmp_path):
    # Regression: the old loader hard-coded sample["jpg"] and silently dropped .jpeg.
    _make_shards(tmp_path, [8], ext="jpeg")
    url = f"{tmp_path}/shard_0_{{0..0}}.tar"
    n, _ = _drain(D.get_webdataset_loader(4, num_workers=0, train=True, shuffle=0, url=url))
    assert n == 8


def test_loader_caps_workers_to_shard_count(monkeypatch):
    # Regression: the val split is a single shard; with 8 workers empty_check aborts.
    captured = {}

    def fake_loader(dataset, **kw):
        captured.update(kw)
        return dataset

    monkeypatch.setattr(wds, "WebLoader", fake_loader)
    D.get_webdataset_loader(4, num_workers=8, train=False, url="x/shard_0_{0..0}.tar")
    assert captured["num_workers"] == 1
    D.get_webdataset_loader(4, num_workers=8, train=True, url="x/shard_0_{0..49}.tar")
    assert captured["num_workers"] == 8
