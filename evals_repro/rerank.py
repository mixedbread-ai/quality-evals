import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Protocol

import cohere
import voyageai
from mixedbread import Mixedbread

from evals_repro.clients import cohere_v2
from evals_repro.data import Query, Subset
from evals_repro.index import Run
from evals_repro.local_rerank import LOCAL_MODELS, LocalReranker
from evals_repro.retrievers import Retriever


class Reranker(Protocol):
    name: str
    provider: str

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def scores_by_position(results, size: int) -> list[float]:
    scores = [float("nan")] * size
    seen = set()
    for result in results:
        index = result["index"] if isinstance(result, dict) else result.index
        value = result["relevance_score"] if isinstance(result, dict) else result.relevance_score
        if not isinstance(index, int) or not 0 <= index < size or index in seen or not math.isfinite(value):
            raise ValueError("Invalid reranker score or document index")
        seen.add(index)
        scores[index] = float(value)
    if len(seen) != size:
        raise ValueError("Incomplete reranker response")
    return scores


@dataclass
class CohereReranker:
    provider: ClassVar[str] = "cohere"
    model: str = "rerank-v4.0-pro"
    max_tokens_per_doc: int = 8192
    client: cohere.ClientV2 = field(default_factory=cohere_v2)
    retries: int = 2

    @property
    def name(self) -> str:
        return f"cohere-{self.model}"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            max_tokens_per_doc=self.max_tokens_per_doc,
            request_options={"max_retries": self.retries},
        )
        return scores_by_position(response.results, len(documents))


@dataclass
class VoyageReranker:
    provider: ClassVar[str] = "voyage"
    model: str = "rerank-3"
    client: voyageai.Client = field(default_factory=lambda: voyageai.Client(max_retries=0, timeout=120))

    @property
    def name(self) -> str:
        return f"voyage-{self.model}"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            response = self.client.rerank(model=self.model, query=query, documents=documents, truncation=False)
        except voyageai.error.InvalidRequestError as error:
            if "max allowed tokens per submitted batch" not in str(error).lower() or len(documents) < 2:
                raise
            split = len(documents) // 2
            print(f"{self.name}: token overflow; splitting {len(documents)} documents into two API batches", flush=True)
            return self.rerank(query, documents[:split]) + self.rerank(query, documents[split:])
        return scores_by_position(response.results, len(documents))


@dataclass
class MixedbreadReranker:
    provider: ClassVar[str] = "mixedbread"
    model: str = "mixedbread-ai/mxbai-rerank-v3.1-listwise"
    client: Mixedbread = field(default_factory=lambda: Mixedbread(max_retries=0, timeout=120))
    name: str = "mixedbread_prod_260910"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = self.client.rerank(model=self.model, query=query, input=documents, top_k=len(documents))
        return scores_by_position(
            [{"index": r.index, "relevance_score": r.score} for r in response.data], len(documents)
        )


RERANKERS = {
    "cohere": CohereReranker,
    "cohere-pro": partial(CohereReranker, retries=0, max_tokens_per_doc=32768),
    "cohere-fast": partial(CohereReranker, model="rerank-v4.0-fast", retries=0, max_tokens_per_doc=32768),
    "voyage": VoyageReranker,
    "mixedbread": MixedbreadReranker,
    **{key: partial(LocalReranker, key) for key in LOCAL_MODELS},
}


@dataclass
class RerankedRetriever:
    first_stage: Retriever
    reranker: Reranker
    depth: int = 50
    workers: int = 8

    @property
    def name(self) -> str:
        return f"{self.first_stage.name}+{self.reranker.name}@{self.depth}"

    def rerank_one(self, subset: Subset, query: Query, candidates: dict[str, float]) -> dict[str, float]:
        kept = [cid for cid in candidates if subset.pages[cid].strip()]
        if not kept:
            return {}
        scores = self.reranker.rerank(query.text, [subset.pages[cid] for cid in kept])
        return dict(zip(kept, scores, strict=True))

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run:
        first = self.first_stage.run(subset, queries, self.depth)
        with ThreadPoolExecutor(self.workers) as pool:
            reranked = list(pool.map(lambda q: self.rerank_one(subset, q, first[q.id]), queries))
        return {q.id: scores for q, scores in zip(queries, reranked, strict=True)}
