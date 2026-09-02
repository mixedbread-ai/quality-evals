from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np

from evals_repro.cache import embed_cached
from evals_repro.data import Content, Query, Subset
from evals_repro.embedders import Embedder
from evals_repro.index import ExactIndex, Run


class Retriever(Protocol):
    name: str

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run: ...


@dataclass
class DenseRetriever:
    embedder: Embedder
    cache_dir: Path
    content: Content = "markdown"
    indexes: dict[str, ExactIndex] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.embedder.name if self.content == "markdown" else f"{self.embedder.name}-images"

    def cache_path(self, subset: Subset, part: str) -> Path:
        return self.cache_dir / self.name / f"{subset.name}.{part}.npz"

    def page_vectors(self, subset: Subset) -> np.ndarray:
        path = self.cache_path(subset, "documents")
        if self.content == "markdown":
            return embed_cached(partial(self.embedder.embed, kind="document"), list(subset.pages.values()), path)
        return embed_cached(self.embedder.embed_images, subset.page_images(), path)

    def index(self, subset: Subset) -> ExactIndex:
        if subset.name not in self.indexes:
            self.indexes[subset.name] = ExactIndex(list(subset.pages), self.page_vectors(subset))
        return self.indexes[subset.name]

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run:
        language = "+".join(sorted({q.language for q in queries}))
        embed = partial(self.embedder.embed, kind="query")
        vectors = embed_cached(embed, [q.text for q in queries], self.cache_path(subset, f"{language}.queries"))
        return self.index(subset).run([q.id for q in queries], vectors, top_k)
