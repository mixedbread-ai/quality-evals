from functools import partial

import numpy as np
import pytest

from evals_repro.cache import embed_cached
from evals_repro.embedders import batches_by_budget, embed_batched, embed_nonblank
from evals_repro.embedders.images import data_url, png_pixels


class CountingEmbedder:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return np.array([[len(t), kind == "query"] for t in texts], dtype=np.float32)


def test_blank_texts_become_zero_vectors():
    embedded = embed_nonblank(lambda ts: np.ones((len(ts), 3)), ["a", "", "  ", "b"])
    assert embedded.tolist() == [[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]]
    with pytest.raises(ValueError):
        embed_nonblank(lambda ts: np.ones((len(ts), 3)), ["", " "])


def test_batches_respect_size_and_item_limits():
    assert batches_by_budget([50, 60, 10, 10, 10, 10, 90], max_size=100, max_items=3) == [[0], [1, 2, 3], [4, 5], [6]]
    assert batches_by_budget([500], max_size=100, max_items=3) == [[0]]
    assert batches_by_budget([], max_size=100, max_items=3) == []


def test_embed_batched_preserves_order():
    request = lambda items: np.array([[i] for i in items], dtype=np.float32)  # noqa: E731
    out = embed_batched(request, list(range(7)), [1] * 7, max_size=2, max_items=3, workers=3)
    assert out[:, 0].tolist() == list(range(7))


def test_cache_hits_only_on_identical_items(tmp_path):
    embedder = CountingEmbedder()
    embed = partial(embedder.embed, kind="query")
    path = tmp_path / "e.npz"
    first = embed_cached(embed, ["ab", "c"], path)
    again = embed_cached(embed, ["ab", "c"], path)
    assert embedder.calls == 1 and np.array_equal(first, again)
    embed_cached(embed, ["ab", "d"], path)
    assert embedder.calls == 2
    embed_cached(lambda items: np.ones((len(items), 1)), [b"\x89PNG", b"raw"], path)
    assert embed_cached(lambda items: np.zeros((len(items), 1)), [b"\x89PNG", b"raw"], path).sum() == 2


def test_png_header_pixels():
    import struct

    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1000, 1300) + b"\x00" * 5
    assert png_pixels(png) == 1_300_000
    assert data_url(png).startswith("data:image/png;base64,iVBOR")
