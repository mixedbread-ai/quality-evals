from types import SimpleNamespace

import pytest

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


def test_scores_by_position_rejects_incomplete_responses():
    with pytest.raises(ValueError, match="Incomplete"):
        scores_by_position([SimpleNamespace(index=1, relevance_score=0.4)], 3)


def test_budget_sleeps_until_window_frees(monkeypatch):
    clock = [0.0]
    slept = []
    monkeypatch.setattr("evals_repro.throttle.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "evals_repro.throttle.time.sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s))
    )
    budget = Budget(100)
    monkeypatch.setattr(budget.ready, "wait", lambda t: (slept.append(t), clock.__setitem__(0, clock[0] + t)))
    budget.reserve(60)
    budget.reserve(40)
    budget.reserve(10)
    assert slept and abs(slept[0] - 60) < 1e-9


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 2}],
        [{"index": 0, "relevance_score": 1}, {"index": 2, "relevance_score": 2}],
        [{"index": 0, "relevance_score": 1}, {"index": 1, "relevance_score": float("nan")}],
    ],
)
def test_score_indices_and_values_are_validated(results):
    with pytest.raises(ValueError, match="Invalid"):
        scores_by_position(results, 2)


def test_voyage_splits_only_batch_overflow_without_changing_documents():
    import voyageai

    from evals_repro.rerank import VoyageReranker

    calls = []

    def rerank(**kwargs):
        docs = kwargs["documents"]
        calls.append(docs)
        assert kwargs["truncation"] is False
        if len(docs) > 2:
            raise voyageai.error.InvalidRequestError("max allowed tokens per submitted batch", None)
        return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=float(d)) for i, d in enumerate(docs)])

    runner = VoyageReranker(client=SimpleNamespace(rerank=rerank))
    assert runner.rerank("q", ["1", "2", "3", "4"]) == [1, 2, 3, 4]
    assert calls == [["1", "2", "3", "4"], ["1", "2"], ["3", "4"]]


def test_cohere_study_disables_sdk_retries_and_requests_32k_documents():
    from evals_repro.rerank import RERANKERS

    def rerank(**kwargs):
        assert kwargs["max_tokens_per_doc"] == 32768
        assert kwargs["request_options"] == {"max_retries": 0}
        return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.8)])

    model = RERANKERS["cohere-pro"](client=SimpleNamespace(rerank=rerank))
    assert model.rerank("q", ["doc"]) == [0.8]
