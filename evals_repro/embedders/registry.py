from functools import partial

from evals_repro.embedders import Embedder
from evals_repro.embedders.cohere import CohereEmbedder
from evals_repro.embedders.gemini import BARE, GeminiEmbedder
from evals_repro.embedders.voyage import VoyageEmbedder
from evals_repro.embedders.voyage_multimodal import VoyageMultimodalEmbedder

EMBEDDERS: dict[str, type[Embedder] | partial] = {
    "voyage-4-large": partial(VoyageEmbedder, "voyage-4-large"),
    "voyage-4": partial(VoyageEmbedder, "voyage-4"),
    "voyage-4-lite": partial(VoyageEmbedder, "voyage-4-lite"),
    "voyage-multimodal-3.5": VoyageMultimodalEmbedder,
    "cohere-embed-v4.0": partial(CohereEmbedder, "embed-v4.0"),
    "gemini-embedding-2": GeminiEmbedder,
    "gemini-embedding-2-noprompt": partial(GeminiEmbedder, prompts=BARE),
}
