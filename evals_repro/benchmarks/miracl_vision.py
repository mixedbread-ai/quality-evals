from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from evals_repro.benchmarks import Benchmark
from evals_repro.data import Query, Subset, check_consistency

REPO_ID = "nvidia/miracl-vision"
LANGUAGE_NAMES = {
    "ar": "arabic",
    "bn": "bengali",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fa": "farsi",
    "fi": "finnish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "ja": "japanese",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",
    "th": "thai",
    "yo": "yoruba",
    "zh": "chinese",
}


def snapshot(language: str) -> Path:
    patterns = [f"{language}/*.parquet"]
    return Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=patterns)) / language


def read_rows(root: Path, name: str, columns: list[str]) -> list[dict]:
    return pq.read_table(root / f"{name}.parquet", columns=columns).to_pylist()


def load_qrels(root: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in read_rows(root, "qrels", ["query-id", "corpus-id", "score"]):
        if row["score"] > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = row["score"]
    return qrels


def load_queries(root: Path, qrels: dict[str, dict[str, int]], language: str) -> list[Query]:
    rows = read_rows(root, "queries", ["_id", "text"])
    return [Query(r["_id"], r["text"], language) for r in rows if r["_id"] in qrels]


def load_corpus(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    rows = sorted(read_rows(root, "corpus", ["_id", "title", "text", "image_id"]), key=lambda r: int(r["_id"]))
    titles = {r["_id"]: r["title"] or "" for r in rows}
    pages = {cid: f"title: {titles[cid]}\n{r['text'] or ''}" for cid, r in zip(titles, rows, strict=True)}
    page_of_image = {r["image_id"]: r["_id"] for r in rows}
    assert len(page_of_image) == len(pages), f"{root.name}: pages share images"
    return pages, titles, page_of_image


def iter_page_images(root: Path, page_of_image: dict[str, str], batch_size: int = 64) -> Iterator[tuple[str, bytes]]:
    file = pq.ParquetFile(root / "images.parquet")
    assert file.metadata.num_rows == len(page_of_image), f"{root.name}: images and pages disagree"
    for batch in file.iter_batches(batch_size, columns=["file_name", "image"]):
        for row in batch.to_pylist():
            yield page_of_image[row["file_name"]], row["image"]["bytes"]


def build_subset(language: str, root: Path) -> Subset:
    qrels = load_qrels(root)
    pages, titles, page_of_image = load_corpus(root)
    native = LANGUAGE_NAMES[language]
    loaded = Subset(
        language,
        f"{REPO_ID}/{language}",
        native,
        pages,
        load_queries(root, qrels, native),
        qrels,
        partial(iter_page_images, root, page_of_image),
        titles,
    )
    check_consistency(loaded)
    return loaded


def load_subset(language: str) -> Subset:
    return build_subset(language, snapshot(language))


BENCHMARK = Benchmark("miracl-vision", tuple(LANGUAGE_NAMES), (), load_subset)
