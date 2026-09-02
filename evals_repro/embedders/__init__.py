from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol

import numpy as np

Kind = Literal["query", "document"]
CHARS_PER_TOKEN = 3


def caption(title: str) -> str:
    return f"title: {title}"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray: ...


def embed_nonblank(embed: Callable[[list[str]], np.ndarray], texts: list[str]) -> np.ndarray:
    keep = [i for i, t in enumerate(texts) if t.strip()]
    if not keep:
        raise ValueError("nothing to embed")
    vectors = embed([texts[i] for i in keep])
    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    out[keep] = vectors
    return out


def batches_by_budget(sizes: Sequence[int], max_size: int, max_items: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, size in enumerate(sizes):
        if current and (used + size > max_size or len(current) >= max_items):
            groups.append(current)
            current, used = [], 0
        current.append(i)
        used += size
    if current:
        groups.append(current)
    return groups


def embed_batched(
    request: Callable[[list], np.ndarray],
    items: Sequence,
    sizes: Sequence[int],
    max_size: int,
    max_items: int,
    workers: int,
) -> np.ndarray:
    groups = batches_by_budget(sizes, max_size, max_items)
    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(lambda g: request([items[i] for i in g]), groups)))
