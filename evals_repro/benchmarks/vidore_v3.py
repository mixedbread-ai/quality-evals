from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from evals_repro.benchmarks import Benchmark
from evals_repro.data import Query, Subset, check_consistency

NATIVE_LANGUAGE = {
    "computer_science": "english",
    "energy": "french",
    "finance_en": "english",
    "finance_fr": "french",
    "hr": "english",
    "industrial": "english",
    "pharmaceuticals": "english",
    "physics": "french",
}
LANGUAGES = ("english", "french", "spanish", "italian", "german", "portuguese")


def repo_id(subset: str) -> str:
    return f"vidore/vidore_v3_{subset}"


def snapshot(subset: str) -> Path:
    patterns = ["corpus/*.parquet", "queries/*.parquet", "qrels/*.parquet"]
    return Path(snapshot_download(repo_id(subset), repo_type="dataset", allow_patterns=patterns))


def read_rows(root: Path, config: str, columns: list[str]) -> list[dict]:
    return pq.read_table(root / config, columns=columns).to_pylist()


def load_qrels(root: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in read_rows(root, "qrels", ["query_id", "corpus_id", "score"]):
        if row["score"] > 0:
            qrels.setdefault(str(row["query_id"]), {})[str(row["corpus_id"])] = row["score"]
    return qrels


def load_queries(root: Path, qrels: dict[str, dict[str, int]]) -> list[Query]:
    rows = read_rows(root, "queries", ["query_id", "query", "language"])
    queries = [Query(str(r["query_id"]), r["query"], r["language"]) for r in rows]
    return [q for q in queries if q.id in qrels]


def load_pages(root: Path) -> dict[str, str]:
    rows = read_rows(root, "corpus", ["corpus_id", "markdown"])
    return {str(r["corpus_id"]): r["markdown"] or "" for r in sorted(rows, key=lambda r: r["corpus_id"])}


def iter_page_images(root: Path, batch_size: int = 64) -> Iterator[tuple[str, bytes]]:
    for batch in pq.ParquetDataset(root / "corpus").read(columns=["corpus_id", "image"]).to_batches(batch_size):
        for row in batch.to_pylist():
            yield str(row["corpus_id"]), row["image"]["bytes"]


def load_subset(subset: str) -> Subset:
    root = snapshot(subset)
    qrels = load_qrels(root)
    loaded = Subset(
        subset,
        repo_id(subset),
        NATIVE_LANGUAGE[subset],
        load_pages(root),
        load_queries(root, qrels),
        qrels,
        partial(iter_page_images, root),
    )
    check_consistency(loaded)
    return loaded


BENCHMARK = Benchmark("vidore-v3", tuple(NATIVE_LANGUAGE), LANGUAGES, load_subset)
