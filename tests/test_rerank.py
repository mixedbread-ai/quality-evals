from types import SimpleNamespace

from evals_repro.data import Query, Subset
from evals_repro.rerank import RerankedRetriever, scores_by_position
from evals_repro.throttle import Budget


class Reverser:
    name = "reverser"

    def rerank(self, query, documents):
        return [float(-len(d)) for d in documents]


class Constant:
    name = "constant"

    def __init__(self, hits):
        self.hits = hits
        self.depths = []

    def run(self, subset, queries, top_k):
        self.depths.append(top_k)
        return {q.id: dict(self.hits) for q in queries}


def test_reranker_rescores_first_stage_and_drops_blank_pages():
    subset = Subset("hr", "src", "english", {"0": "long text", "1": "", "2": "hi"}, [], {})
    first = Constant({"0": 0.9, "1": 0.8, "2": 0.7})
    retriever = RerankedRetriever(first, Reverser(), depth=3)
    run = retriever.run(subset, [Query("q", "t", "english")], 10)
    assert run == {"q": {"0": -9.0, "2": -2.0}}
    assert first.depths == [3] and retriever.name == "constant+reverser@3"


def test_empty_first_stage_skips_reranker():
    class Explodes:
        name = "x"

        def rerank(self, query, documents):
            raise AssertionError

    run = RerankedRetriever(Constant({}), Explodes()).run(
        Subset("hr", "src", "english", {}, [], {}), [Query("q", "t", "english")], 10
    )
    assert run == {"q": {}}


def test_scores_by_position_fills_missing_with_minus_inf():
    assert scores_by_position([SimpleNamespace(index=1, relevance_score=0.4)], 3) == [float("-inf"), 0.4, float("-inf")]


def test_budget_sleeps_until_window_frees(monkeypatch):
    clock = [0.0]
    slept = []
    monkeypatch.setattr("evals_repro.throttle.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "evals_repro.throttle.time.sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s))
    )
    budget = Budget(100)
    budget.reserve(60)
    budget.reserve(40)
    budget.reserve(10)
    assert slept and abs(slept[0] - 60) < 1e-9
