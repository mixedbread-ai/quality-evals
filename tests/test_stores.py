from types import SimpleNamespace

from evals_repro.data import Query, Subset
from evals_repro.stores import StoreRetriever, StoreUploader, file_scores, page_files, store_name


def test_store_names_are_valid_identifiers():
    assert store_name("computer_science", "markdown", "vidore-eval") == "vidore-eval-computer-science-markdown"


def test_markdown_files_are_named_by_corpus_id():
    files = dict(page_files(Subset("hr", "src", "english", {"0": "hello", "1": ""}, [], {}), "markdown"))
    assert files["0"] == ("0.md", b"hello", "text/markdown")
    assert files["1"][1] == b""


def test_chunks_collapse_to_best_score_per_page():
    hits = [
        SimpleNamespace(external_id="7", filename="7.md", score=0.5),
        SimpleNamespace(external_id="7", filename="7.md", score=0.9),
        SimpleNamespace(external_id=None, filename="12.png", score=0.7),
    ]
    assert file_scores(hits) == {"7": 0.9, "12": 0.7}


def test_retriever_names_and_options():
    plain = StoreRetriever("p", "markdown", client=None)
    reranked = StoreRetriever("p", "images", rerank_top_k=10, client=None)
    assert plain.name == "mixedbread-markdown" and plain.search_options() == {"rerank": False}
    assert reranked.name == "mixedbread-images+mxbai-rerank-v3.1-listwise"
    assert reranked.search_options()["rerank"]["top_k"] == 10


def test_run_maps_queries_to_pages():
    class Stores:
        def search(self, *, query, store_identifiers, top_k, search_options):
            return SimpleNamespace(data=[SimpleNamespace(external_id="3", filename="3.md", score=float(len(query)))])

    retriever = StoreRetriever("p", "markdown", client=SimpleNamespace(stores=Stores()))
    run = retriever.run(
        Subset("hr", "src", "english", {"3": "x"}, [], {}),
        [Query("q1", "ab", "english"), Query("q2", "abc", "english")],
        10,
    )
    assert run == {"q1": {"3": 2.0}, "q2": {"3": 3.0}}


def test_upload_skips_blank_and_existing_pages():
    uploaded = []

    class Files:
        def upload(self, **kwargs):
            uploaded.append(kwargs["external_id"])

        def list(self, name, limit, after):
            data = [SimpleNamespace(external_id="1", status="completed")]
            return SimpleNamespace(data=data, pagination=SimpleNamespace(has_more=False, last_cursor=None))

    class Stores:
        files = Files()

        def list(self, limit):
            return [SimpleNamespace(name="vidore-eval-hr-markdown")]

    uploader = StoreUploader("vidore-eval", client=SimpleNamespace(stores=Stores()))
    subset = Subset("hr", "src", "english", {"0": "a", "1": "b", "2": " "}, [], {})
    assert uploader.upload(subset, "markdown") == "vidore-eval-hr-markdown"
    assert uploaded == ["0"]
