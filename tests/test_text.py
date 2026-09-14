from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from evals_repro.benchmarks import text


def test_beir_uses_test_qrels_and_title_plus_body(tmp_path, monkeypatch):
    for part, rows in {
        "corpus": [{"_id": "a", "title": "Title", "text": "Body"}, {"_id": "b", "title": "", "text": "Other"}],
        "queries": [{"_id": "q", "text": "question"}, {"_id": "unused", "text": "unjudged"}],
    }.items():
        (tmp_path / part).mkdir()
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / part / "data.parquet")
    qrels = tmp_path / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq\ta\t2\nq\tb\t0\n")
    monkeypatch.setattr(text, "beir_snapshot", lambda name: tmp_path)
    monkeypatch.setattr(text, "hf_hub_download", lambda repo, filename, **kw: str(qrels))
    text.corpus.cache_clear()
    subset = text.load_subset("fiqa")
    assert subset.pages == {"a": "Title\nBody", "b": "Other"}
    assert [q.id for q in subset.queries] == ["q"]
    assert subset.qrels == {"q": {"a": 2}} and subset.corpus_id == "beir/fiqa"
    text.corpus.cache_clear()


def test_dl_tasks_share_the_full_passage_corpus(monkeypatch):
    loaded = []

    def dataset(source):
        loaded.append(source)
        return SimpleNamespace(
            docs_iter=lambda: iter([SimpleNamespace(doc_id="d", text="passage")]),
            queries_iter=lambda: iter([SimpleNamespace(query_id="q", text="question")]),
            qrels_iter=lambda: iter([SimpleNamespace(query_id="q", doc_id="d", relevance=3)]),
        )

    monkeypatch.setattr(text.ir_datasets, "load", dataset)
    text.corpus.cache_clear()
    first, second = text.load_subset("dl19"), text.load_subset("dl20")
    assert first.pages is second.pages
    assert loaded.count("msmarco-passage") == 1
    assert first.qrels == {"q": {"d": 3}}
    text.corpus.cache_clear()
