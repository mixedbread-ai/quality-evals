from dataclasses import dataclass, field
from functools import partial

import numpy as np
import voyageai

from evals_repro.clients import voyage
from evals_repro.data import TitledImage
from evals_repro.embedders import CHARS_PER_TOKEN, Kind, caption, embed_batched, embed_nonblank
from evals_repro.embedders.images import data_url, png_pixels
from evals_repro.throttle import Budget

PIXELS_PER_TOKEN = 560
MAX_PIXELS = 2_000_000


def text_input(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def image_input(title: str, png: bytes) -> dict:
    text = [{"type": "text", "text": caption(title)}] if title else []
    return {"content": [*text, {"type": "image_base64", "image_base64": data_url(png)}]}


def image_tokens(title: str, png: bytes) -> int:
    return len(title) // CHARS_PER_TOKEN + min(png_pixels(png), MAX_PIXELS) // PIXELS_PER_TOKEN


@dataclass
class VoyageMultimodalEmbedder:
    model: str = "voyage-multimodal-3.5"
    max_batch_tokens: int = 200_000
    max_batch_items: int = 64
    workers: int = 4
    budget: Budget = field(default_factory=lambda: Budget(1_500_000))
    client: voyageai.Client = field(default_factory=voyage)

    @property
    def name(self) -> str:
        return self.model

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        def embed_all(kept: list[str]) -> np.ndarray:
            sizes = [len(t) // CHARS_PER_TOKEN + 1 for t in kept]
            return self.embed_inputs([text_input(t) for t in kept], sizes, kind)

        return embed_nonblank(embed_all, texts)

    def embed_images(self, images: list[TitledImage]) -> np.ndarray:
        return self.embed_inputs([image_input(*i) for i in images], [image_tokens(*i) for i in images], "document")

    def embed_inputs(self, inputs: list[dict], sizes: list[int], kind: Kind) -> np.ndarray:
        request = partial(self.request, kind=kind)
        return embed_batched(
            request,
            list(zip(inputs, sizes, strict=True)),
            sizes,
            self.max_batch_tokens,
            self.max_batch_items,
            self.workers,
        )

    def request(self, batch: list[tuple[dict, int]], kind: Kind) -> np.ndarray:
        inputs, sizes = zip(*batch, strict=True)
        self.budget.reserve(sum(sizes))
        response = self.client.multimodal_embed(list(inputs), model=self.model, input_type=kind, truncation=True)
        return np.asarray(response.embeddings, dtype=np.float32)
