from dataclasses import dataclass, field
from functools import partial

import numpy as np
import voyageai

from evals_repro.clients import voyage
from evals_repro.embedders import Kind, embed_batched, embed_nonblank


@dataclass
class VoyageEmbedder:
    model: str = "voyage-4-large"
    max_batch_tokens: int = 100_000
    max_batch_items: int = 256
    workers: int = 4
    client: voyageai.Client = field(default_factory=voyage)

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        return embed_nonblank(partial(self.embed_all, kind=kind), texts)

    def embed_all(self, texts: list[str], kind: Kind) -> np.ndarray:
        sizes = [len(t) for t in self.client.tokenize(texts, self.model)]
        request = partial(self.request, kind=kind)
        return embed_batched(request, texts, sizes, self.max_batch_tokens, self.max_batch_items, self.workers)

    def request(self, texts: list[str], kind: Kind) -> np.ndarray:
        response = self.client.embed(texts, model=self.model, input_type=kind, truncation=True)
        return np.asarray(response.embeddings, dtype=np.float32)
