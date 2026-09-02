import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

Item = str | bytes | tuple[str | bytes, ...]


def digest(items: Sequence[Item]) -> str:
    h = hashlib.sha256()
    for item in items:
        for part in item if isinstance(item, tuple) else (item,):
            h.update(part if isinstance(part, bytes) else part.encode())
            h.update(b"\0")
        h.update(b"\1")
    return h.hexdigest()


def embed_cached(embed: Callable[[list], np.ndarray], items: Sequence[Item], path: Path) -> np.ndarray:
    key = digest(items)
    if path.exists():
        with np.load(path) as saved:
            if str(saved["digest"]) == key:
                return saved["vectors"]
    vectors = embed(list(items))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, digest=key)
    return vectors
