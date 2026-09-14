import csv
from functools import lru_cache
from pathlib import Path

import ir_datasets
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from evals_repro.benchmarks import Benchmark
from evals_repro.data import Query, Subset, check_consistency

DATASETS = {
    "fiqa": "beir/fiqa/test",
    "trec-covid": "beir/trec-covid",
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
    "hotpotqa": "beir/hotpotqa/test",
    "nq": "beir/nq",
    "scifact": "beir/scifact/test",
}


REVISIONS = {
    "fiqa": ("979c07a7cb5ccc6ca009792241fa1250b98055dd", "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"),
    "trec-covid": ("7e16fde3016c639c7f856e803f4bab92645562c4", "532ac68ee6756ac22c9346eebf65bd3c6a042e10"),
    "hotpotqa": ("a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b", "b15429e9244c8ec966985d7778427c3b1543b314"),
    "nq": ("b7253e6c379163d024ddb1d6948152a91a2e3b46", "519acd4e48bb3e5da22b2b888ce36c614f4f2bc9"),
    "scifact": ("b3b5335604bf5ee3c4447671af975ea25143d4f5", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"),
}


@lru_cache(maxsize=1)
def corpus(source: str) -> dict[str, str]:
    if source == "msmarco-passage":
        return {doc.doc_id: doc.text for doc in ir_datasets.load(source).docs_iter()}
    root = beir_snapshot(source.split("/")[1])
    pages = {}
    for path in sorted((root / "corpus").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["_id", "title", "text"]):
            for doc in batch.to_pylist():
                pages[str(doc["_id"])] = "\n".join(filter(None, (doc["title"], doc["text"]))).strip()
    return pages


def beir_snapshot(name: str) -> Path:
    return Path(
        snapshot_download(
            f"BeIR/{name}",
            revision=REVISIONS[name][0],
            repo_type="dataset",
            allow_patterns=["corpus/*.parquet", "queries/*.parquet"],
        )
    )


def beir_queries(name: str) -> tuple[list[Query], dict]:
    path = hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", revision=REVISIONS[name][1], repo_type="dataset")
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    rows = pq.read_table(beir_snapshot(name) / "queries", columns=["_id", "text"]).to_pylist()
    return [Query(str(q["_id"]), q["text"], "english") for q in rows if str(q["_id"]) in qrels], qrels


def load_subset(name: str) -> Subset:
    source = DATASETS[name]
    if source.startswith("beir/"):
        queries, qrels = beir_queries(name)
    else:
        dataset = ir_datasets.load(source)
        qrels = {}
        for rel in dataset.qrels_iter():
            if rel.relevance > 0:
                qrels.setdefault(rel.query_id, {})[rel.doc_id] = rel.relevance
        queries = [Query(q.query_id, q.text, "english") for q in dataset.queries_iter() if q.query_id in qrels]
    corpus_id = "msmarco-passage" if source.startswith("msmarco-passage/") else "/".join(source.split("/")[:2])
    source = source + ("@" + "+".join(REVISIONS[name]) if name in REVISIONS else "")
    subset = Subset(name, source, "english", corpus(corpus_id), queries, qrels, corpus_id=corpus_id)
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("text", tuple(DATASETS), (), load_subset)
