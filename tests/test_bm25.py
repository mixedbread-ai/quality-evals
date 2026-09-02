from evals_repro.bm25 import BM25Retriever
from evals_repro.data import Query, Subset


def subset():
    pages = {
        "0": "python variables and naming rules",
        "1": "bread baking with sourdough",
        "2": "sourdough starter care",
    }
    return Subset("computer_science", "src", "english", pages, [], {})


def test_lexical_match_ranks_first():
    run = BM25Retriever().run(subset(), [Query("q", "sourdough bread", "english")], top_k=10)
    assert max(run["q"], key=run["q"].get) == "1" and "0" not in run["q"]


def test_no_overlap_returns_nothing():
    assert BM25Retriever().run(subset(), [Query("q", "quantum", "english")], top_k=10) == {"q": {}}
