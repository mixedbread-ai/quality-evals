from dataclasses import dataclass, field

import bm25s
import Stemmer

from evals_repro.data import Query, Subset
from evals_repro.index import Run


def tokens(texts: list[str], stemmer: Stemmer.Stemmer) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords=None, stemmer=stemmer, show_progress=False)


@dataclass
class BM25Index:
    retriever: bm25s.BM25
    stemmer: Stemmer.Stemmer
    ids: list[str]

    @classmethod
    def build(cls, subset: Subset) -> "BM25Index":
        stemmer = Stemmer.Stemmer(subset.native_language)
        retriever = bm25s.BM25()
        retriever.index(tokens(list(subset.pages.values()), stemmer), show_progress=False)
        return cls(retriever, stemmer, list(subset.pages))

    def search(self, texts: list[str], top_k: int) -> tuple:
        return self.retriever.retrieve(tokens(texts, self.stemmer), k=min(top_k, len(self.ids)), show_progress=False)


@dataclass
class BM25Retriever:
    name: str = "bm25"
    indexes: dict[str, BM25Index] = field(default_factory=dict)

    def index(self, subset: Subset) -> BM25Index:
        if subset.name not in self.indexes:
            self.indexes[subset.name] = BM25Index.build(subset)
        return self.indexes[subset.name]

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run:
        index = self.index(subset)
        hits, scores = index.search([q.text for q in queries], top_k)
        return {
            q.id: {index.ids[h]: float(s) for h, s in zip(row_hits, row_scores, strict=True) if s > 0}
            for q, row_hits, row_scores in zip(queries, hits, scores, strict=True)
        }
