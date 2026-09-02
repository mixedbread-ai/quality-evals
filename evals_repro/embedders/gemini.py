from dataclasses import dataclass, field

import numpy as np
from google import genai
from google.genai import types

from evals_repro.clients import gemini
from evals_repro.data import TitledImage
from evals_repro.embedders import Kind, caption, embed_batched, embed_nonblank

PROMPTS = {"query": "task: search result | query: {}", "document": "title: none | text: {}"}
BARE = {"query": "{}", "document": "{}"}


def text_content(text: str) -> types.Content:
    return types.Content(parts=[types.Part.from_text(text=text)])


def image_content(title: str, png: bytes) -> types.Content:
    text = [types.Part.from_text(text=caption(title))] if title else []
    return types.Content(parts=[*text, types.Part.from_bytes(data=png, mime_type="image/png")])


@dataclass
class GeminiEmbedder:
    model: str = "gemini-embedding-2"
    prompts: dict[Kind, str] = field(default_factory=lambda: dict(PROMPTS))
    max_batch_items: int = 50
    max_batch_chars: int = 400_000
    workers: int = 4
    client: genai.Client = field(default_factory=gemini)

    @property
    def name(self) -> str:
        return self.model if self.prompts == PROMPTS else f"{self.model}-noprompt"

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        def embed_all(kept: list[str]) -> np.ndarray:
            contents = [text_content(self.prompts[kind].format(t)) for t in kept]
            sizes = [len(t) for t in kept]
            return embed_batched(
                self.request, contents, sizes, self.max_batch_chars, self.max_batch_items, self.workers
            )

        return embed_nonblank(embed_all, texts)

    def embed_images(self, images: list[TitledImage]) -> np.ndarray:
        contents = [image_content(*i) for i in images]
        return embed_batched(
            self.request, contents, [1] * len(images), self.max_batch_items, self.max_batch_items, self.workers
        )

    def request(self, contents: list[types.Content]) -> np.ndarray:
        response = self.client.models.embed_content(model=self.model, contents=contents)
        return np.asarray([e.values for e in response.embeddings], dtype=np.float32)
