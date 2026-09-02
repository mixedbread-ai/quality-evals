from dataclasses import dataclass, field
from functools import partial

import cohere
import numpy as np

from evals_repro.clients import cohere_v2
from evals_repro.data import TitledImage
from evals_repro.embedders import Kind, caption, embed_batched, embed_nonblank
from evals_repro.embedders.images import data_url, png_pixels
from evals_repro.throttle import Budget


def image_input(title: str, png: bytes) -> dict:
    text = [{"type": "text", "text": caption(title)}] if title else []
    return {"content": [*text, {"type": "image_url", "image_url": {"url": data_url(png)}}]}


@dataclass
class CohereEmbedder:
    model: str = "embed-v4.0"
    max_batch_items: int = 96
    max_batch_chars: int = 400_000
    max_batch_pixels: int = 45_000_000
    images_per_minute: Budget = field(default_factory=lambda: Budget(380))
    workers: int = 4
    client: cohere.ClientV2 = field(default_factory=cohere_v2)

    @property
    def name(self) -> str:
        return f"cohere-{self.model}"

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        def embed_all(kept: list[str]) -> np.ndarray:
            request = partial(self.request, f"search_{kind}", "texts")
            return embed_batched(
                request, kept, [len(t) for t in kept], self.max_batch_chars, self.max_batch_items, self.workers
            )

        return embed_nonblank(embed_all, texts)

    def embed_images(self, images: list[TitledImage]) -> np.ndarray:
        request = partial(self.request_images, "search_document", "inputs")
        sizes = [png_pixels(png) for _, png in images]
        return embed_batched(
            request,
            [image_input(*i) for i in images],
            sizes,
            self.max_batch_pixels,
            self.max_batch_items,
            self.workers,
        )

    def request_images(self, input_type: str, field: str, inputs: list[dict]) -> np.ndarray:
        self.images_per_minute.reserve(len(inputs))
        return self.request(input_type, field, inputs)

    def request(self, input_type: str, field: str, payload: list) -> np.ndarray:
        response = self.client.embed(
            model=self.model,
            input_type=input_type,
            embedding_types=["float"],
            request_options={"max_retries": 8},
            **{field: payload},
        )
        return np.asarray(response.embeddings.float_, dtype=np.float32)
