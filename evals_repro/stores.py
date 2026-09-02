import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from mixedbread import Mixedbread

from evals_repro.clients import mixedbread
from evals_repro.data import Content, Query, Subset
from evals_repro.index import Run

Parsing = Literal["fast", "high_quality"]
RERANKER = "mixedbread-ai/mxbai-rerank-v3.1-listwise"
File = tuple[str, bytes, str]


def store_name(subset: str, content: Content, prefix: str) -> str:
    return f"{prefix}-{subset.replace('_', '-')}-{content}"


def page_files(subset: Subset, content: Content) -> Iterator[tuple[str, File]]:
    if content == "markdown":
        for cid, text in subset.pages.items():
            yield cid, (f"{cid}.md", text.encode(), "text/markdown")
    else:
        for cid, png in subset.images():
            yield cid, (f"{cid}.png", png, "image/png")


@dataclass
class StoreUploader:
    prefix: str
    parsing: Parsing = "fast"
    workers: int = 8
    client: Mixedbread = field(default_factory=mixedbread)

    def ensure_store(self, subset: Subset, content: Content) -> str:
        name = store_name(subset.name, content, self.prefix)
        if name not in {s.name for s in self.client.stores.list(limit=100)}:
            self.client.stores.create(
                name=name,
                description=f"{subset.name} page {content} from {subset.source}",
                metadata={"subset": subset.name, "content": content, "parsing": self.parsing},
            )
        return name

    def upload(self, subset: Subset, content: Content) -> str:
        name = self.ensure_store(subset, content)
        done = self.uploaded_ids(name)
        todo = ((cid, f) for cid, f in page_files(subset, content) if cid not in done and f[1].strip())
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(lambda cf: self.upload_one(name, subset, *cf), todo))
        return name

    def upload_one(self, store: str, subset: Subset, cid: str, file: File) -> None:
        title = {"title": subset.title(cid)} if subset.title(cid) else {}
        self.client.stores.files.upload(
            store_identifier=store,
            file=file,
            external_id=cid,
            metadata={"corpus_id": cid, "subset": subset.name, **title},
            config={"parsing_strategy": self.parsing},
        )

    def uploaded_ids(self, name: str) -> set[str]:
        ids: set[str] = set()
        after = None
        while True:
            page = self.client.stores.files.list(name, limit=100, after=after)
            ids.update(f.external_id for f in page.data if f.external_id and f.status != "failed")
            if not page.pagination.has_more:
                return ids
            after = page.pagination.last_cursor

    def wait_processed(self, name: str, poll_seconds: float = 5) -> dict[str, int]:
        while True:
            counts = self.client.stores.retrieve(name).file_counts
            if counts.pending + counts.in_progress == 0:
                return {"completed": counts.completed, "failed": counts.failed, "total": counts.total}
            time.sleep(poll_seconds)


def file_scores(hits: Iterable) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hit in hits:
        cid = hit.external_id or hit.filename.rsplit(".", 1)[0]
        scores[cid] = max(scores.get(cid, float("-inf")), hit.score)
    return scores


@dataclass
class StoreRetriever:
    prefix: str
    content: Content
    rerank_top_k: int | None = None
    reranker: str = RERANKER
    workers: int = 16
    client: Mixedbread = field(default_factory=mixedbread)

    @property
    def name(self) -> str:
        tag = "" if self.rerank_top_k is None else f"+{self.reranker.split('/')[-1]}"
        return f"mixedbread-{self.content}{tag}"

    def search_options(self) -> dict:
        if self.rerank_top_k is None:
            return {"rerank": False}
        return {"rerank": {"model": self.reranker, "top_k": self.rerank_top_k}}

    def search(self, store: str, query: Query, top_k: int) -> dict[str, float]:
        response = self.client.stores.search(
            query=query.text, store_identifiers=[store], top_k=top_k, search_options=self.search_options()
        )
        return file_scores(response.data)

    def run(self, subset: Subset, queries: list[Query], top_k: int) -> Run:
        store = store_name(subset.name, self.content, self.prefix)
        with ThreadPoolExecutor(self.workers) as pool:
            scores = list(pool.map(lambda q: self.search(store, q, top_k), queries))
        return {q.id: s for q, s in zip(queries, scores, strict=True)}
