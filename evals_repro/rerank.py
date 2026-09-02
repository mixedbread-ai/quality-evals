from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import cohere

from evals_repro.clients import cohere_v2
from evals_repro.data import Query, Subset
from evals_repro.index import Run
from evals_repro.retrievers import Retriever


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def scores_by_position(results, size: int) -> list[float]:
    scores = [float("-inf")] * size
    for result in results:
        scores[result.index] = result.relevance_score
    return scores


@dataclass
class CohereReranker:
    model: str = "rerank-v4.0-pro"
    max_tokens_per_doc: int = 8192
    client: cohere.ClientV2 = field(default_factory=cohere_v2)

    @property
    def name(self) -> str:
        return f"cohere-{self.model}"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = self.client.rerank(
            model=self.model, query=query, documents=documents, max_tokens_per_doc=self.max_tokens_per_doc
        )
        return scores_by_position(response.results, len(documents))


RERANKERS = {"cohere": CohereReranker}


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
