import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import bm25s
import Stemmer
from filelock import FileLock

from evals_repro.cache import digest
from evals_repro.data import Query, Subset
from evals_repro.index import Run
from evals_repro.records import fingerprint


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
        retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene", backend="numpy")
        retriever.index(tokens(list(subset.pages.values()), stemmer), show_progress=False)
        return cls(retriever, stemmer, list(subset.pages))

    def search(self, texts: list[str], top_k: int) -> tuple:
        return self.retriever.retrieve(
            tokens(texts, self.stemmer), k=min(top_k, len(self.ids)), show_progress=False, backend_selection="numpy"
        )


@dataclass
class BM25Retriever:
    name: str = "bm25"
    indexes: dict[str, BM25Index] = field(default_factory=dict)
    cache_dir: Path | None = None
    include_zero: bool = False

    def index(self, subset: Subset) -> BM25Index:
        key = subset.corpus_id or subset.source
        if key not in self.indexes:
            self.indexes.clear()
            self.indexes[key] = self.load_index(subset, key)
        return self.indexes[key]

    def load_index(self, subset: Subset, key: str) -> BM25Index:
        if self.cache_dir is None:
            return BM25Index.build(subset)
        metadata = {
            "corpus_sha256": digest(subset.pages.items()),
            "language": subset.native_language,
            "bm25s": version("bm25s"),
            "stemmer": version("pystemmer"),
            "stopwords": None,
            "method": "lucene",
            "k1": 1.5,
            "b": 0.75,
            "backend": "numpy",
        }
        path = self.cache_dir / key.replace("/", "--") / fingerprint(metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            marker = path / "metadata.json"
            if marker.exists() and json.loads(marker.read_text()) == metadata:
                return BM25Index(
                    bm25s.BM25.load(path, mmap=True, show_progress=False),
                    Stemmer.Stemmer(subset.native_language),
                    list(subset.pages),
                )
            print(f"Building BM25: {key}, {len(subset.pages):,} documents", flush=True)
            index = BM25Index.build(subset)
            marker.unlink(missing_ok=True)
            index.retriever.save(path, show_progress=False)
            marker.write_text(json.dumps(metadata, indent=2))
            return index

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run:
        index = self.index(subset)
        hits, scores = index.search([q.text for q in queries], top_k)
        return {
            q.id: {
                index.ids[h]: float(s) for h, s in zip(row_hits, row_scores, strict=True) if s > 0 or self.include_zero
            }
            for q, row_hits, row_scores in zip(queries, hits, scores, strict=True)
        }
