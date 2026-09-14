import math
from datetime import UTC, datetime
from pathlib import Path

from evals_repro.cache import digest
from evals_repro.data import Query, Subset
from evals_repro.records import append_record, fingerprint, read_records
from evals_repro.retrievers import Retriever
from evals_repro.stores import StoreRetriever, store_name


def candidates(subset: Subset, queries: list[Query], retriever: Retriever, root: Path, depth: int) -> dict:
    path = root / "candidates" / f"{subset.name}.jsonl"
    rows = read_records(path)
    cached = {row["query_id"]: row for row in rows}
    if len(cached) != len(rows):
        raise ValueError(f"Duplicate candidate records: {path}")
    store = store_name(subset.name, "markdown", retriever.prefix) if isinstance(retriever, StoreRetriever) else None
    corpus_digest = digest(subset.pages.items())
    shared = {}
    for query in queries:
        text = subset.retrieval_text.get(query.id, query.text)
        request = fingerprint([corpus_digest, query.text, text, retriever.name, store, depth])
        if query.id in cached:
            row = cached[query.id]
            if row["request_digest"] != request or row["digest"] != fingerprint([request, list(row["scores"].items())]):
                raise ValueError(f"Stale candidates: {subset.name}/{query.id}; use a new study directory")
        else:
            if text not in shared:
                if store:
                    count = depth
                    while True:
                        hits = retriever.search(store, Query(query.id, text, query.language), count)
                        if len(hits) >= depth or count >= 1000:
                            break
                        count = min(1000, count * 2)
                else:
                    hits = retriever.run(subset, [Query(query.id, text, query.language)], depth)[query.id]
                shared[text] = dict(sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:depth])
            scores = shared[text]
            if len(scores) != depth:
                raise ValueError(f"{subset.name}/{query.id}: expected {depth} candidates, got {len(scores)}")
            if any(cid not in subset.pages or not math.isfinite(score) for cid, score in scores.items()):
                raise ValueError(f"{subset.name}/{query.id}: invalid candidate IDs or scores")
            row = {
                "query_id": query.id,
                "language": query.language,
                "request_digest": request,
                "digest": fingerprint([request, list(scores.items())]),
                "scores": scores,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            append_record(path, row)
            cached[query.id] = row
        if text in shared and shared[text] != row["scores"]:
            raise ValueError(f"{subset.name}/{query.id}: identical retrieval queries have different candidates")
        shared[text] = row["scores"]
    return {q.id: cached[q.id] for q in queries}
