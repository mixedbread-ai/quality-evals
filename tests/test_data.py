import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evals_repro.benchmarks import registry
from evals_repro.benchmarks.vidore_v3 import LANGUAGES, load_pages, load_qrels, load_queries
from evals_repro.data import Query, Subset, check_consistency


def write(root, config, rows):
    (root / config).mkdir()
    pq.write_table(pa.Table.from_pylist(rows), root / config / "test.parquet")


@pytest.fixture
def root(tmp_path):
    write(tmp_path, "qrels", [{"query_id": 0, "corpus_id": 1, "score": 2}, {"query_id": 0, "corpus_id": 0, "score": 0}])
    write(
        tmp_path,
        "queries",
        [{"query_id": 0, "query": "q", "language": "english"}, {"query_id": 1, "query": "x", "language": "french"}],
    )
    write(tmp_path, "corpus", [{"corpus_id": 1, "markdown": "one"}, {"corpus_id": 0, "markdown": None}])
    return tmp_path


def test_loaders_filter_and_order(root):
    qrels = load_qrels(root)
    assert qrels == {"0": {"1": 2}}
    assert load_queries(root, qrels) == [Query("0", "q", "english")]
    assert load_pages(root) == {"0": "", "1": "one"}


def test_registry_exposes_vidore_v3():
    benchmark = registry()["vidore-v3"]
    assert len(benchmark.subsets) == 8 and benchmark.languages == LANGUAGES


def subset(pages, queries, qrels):
    return Subset("hr", "src", "english", pages, queries, qrels)


def test_consistency_checks():
    queries = [Query(str(i), "t", lang) for i, lang in enumerate(LANGUAGES)]
    qrels = {q.id: {"0": 1} for q in queries}
    check_consistency(subset({"0": "x"}, queries, qrels))
    with pytest.raises(AssertionError, match="absent"):
        check_consistency(subset({}, queries, qrels))
    with pytest.raises(AssertionError, match="disagree"):
        check_consistency(subset({"0": "x"}, queries[:-1], qrels))
    with pytest.raises(AssertionError, match="uneven"):
        check_consistency(subset({"0": "x"}, queries + [Query("9", "t", "english")], {**qrels, "9": {"0": 1}}))


def test_page_images_follow_page_order():
    s = Subset("hr", "src", "english", {"0": "a", "1": "b"}, [], {}, lambda: iter([("1", b"y"), ("0", b"x")]))
    assert s.page_images() == [("", b"x"), ("", b"y")]
