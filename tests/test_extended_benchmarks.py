import pytest

from evals_repro.benchmarks.followir import paired_metrics
from evals_repro.benchmarks.freshstack import native_metrics
from evals_repro.data import Query, Subset
from evals_repro.study import candidates
from evals_repro.throttle import Budget


def test_legalragbench_normalizes_ids_without_answer_leakage(monkeypatch):
    from evals_repro.benchmarks import legalragbench

    files = {
        "corpus.jsonl": [{"id": 7, "text": "Released passage"}],
        "qa.jsonl": [{"id": 2, "question": "Expert question", "answer": "Secret answer", "relevant_passage_id": 7}],
    }
    monkeypatch.setattr(legalragbench, "rows", lambda source, filename: files[filename])
    subset = legalragbench.load_subset("legalragbench")
    assert subset.pages == {"7": "Released passage"}
    assert subset.queries == [Query("2", "Expert question", "english")]
    assert subset.qrels == {"2": {"7": 1}}


def test_large_token_reservations_cannot_be_starved(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    clock = [0.0]
    monkeypatch.setattr("evals_repro.throttle.time.monotonic", lambda: clock[0])
    budget = Budget(10)
    budget.reserve(9)
    waiting = {"large": Event(), "small": Event()}
    original_wait = budget.ready.wait

    def wait(timeout=None):
        waiting[current_thread().name].set()
        return original_wait(timeout)

    monkeypatch.setattr(budget.ready, "wait", wait)

    def reserve(name, amount):
        current_thread().name = name
        budget.reserve(amount)

    with ThreadPoolExecutor(2) as pool:
        large = pool.submit(reserve, "large", 9)
        assert waiting["large"].wait(2)
        small = pool.submit(reserve, "small", 1)
        try:
            assert waiting["small"].wait(2)
            assert not large.done() and not small.done()
        finally:
            with budget.ready:
                clock[0] = 60
                budget.ready.notify_all()
        large.result(timeout=2)
        small.result(timeout=2)
    assert [amount for _, amount in budget.spent] == [9, 1]


def test_followir_uses_identical_retrieval_text_for_instruction_pairs(tmp_path):
    queries = [Query("q-og", "question original instruction", "english"), Query("q-changed", "question new", "english")]
    subset = Subset(
        "followir",
        "source",
        "english",
        {"a": "alpha", "b": "beta"},
        queries,
        {},
        retrieval_text=dict.fromkeys((q.id for q in queries), "question"),
    )

    class Retriever:
        name = "bm25"

        def run(self, subset, queries, depth):
            assert queries[0].text == "question"
            return {queries[0].id: {"a": 2, "b": 1}}

    frozen = candidates(subset, queries, Retriever(), tmp_path, 2)
    assert frozen["q-og"]["scores"] == frozen["q-changed"]["scores"]
    subset.retrieval_text["q-og"] = "different"
    with pytest.raises(ValueError, match="Stale candidates"):
        candidates(subset, queries, Retriever(), tmp_path, 2)


def test_followir_rank_changes_missing_documents_and_ties():
    changed = {"q": ["a", "missing"]}
    run = {"q-og": {"a": 3, "b": 2, "c": 1}, "q-changed": {"a": 1, "b": 2, "c": 3}}
    result = paired_metrics(changed, run)
    assert result["q-og"]["p_mrr_top100"] == pytest.approx(1 / 3)
    assert result["q-og"]["changed_document_coverage"] == 0.5
    assert result["q-og"] == result["q-changed"]
    reverse = {"q-og": run["q-changed"], "q-changed": run["q-og"]}
    assert paired_metrics(changed, reverse)["q-og"]["p_mrr_top100"] == pytest.approx(-1 / 3)
    tied = {"q-og": {"a": 1, "b": 1}, "q-changed": {"b": 1, "a": 1}}
    assert paired_metrics({"q": ["a"]}, tied)["q-og"]["p_mrr_top100"] == 0
    with pytest.raises(ValueError, match="identical candidates"):
        paired_metrics(changed, {"q-og": {"a": 1}, "q-changed": {"b": 1}})


def test_freshstack_coverage_counts_nuggets_and_alpha_rewards_diversity():
    nuggets = {"q": {"n1": {"a": 1, "b": 1, "c": 0}, "n2": {"a": 0, "b": 0, "c": 1}}}
    diverse = native_metrics(nuggets, {"q": {"a": 3, "c": 2, "b": 1}})["q"]
    redundant = native_metrics(nuggets, {"q": {"a": 3, "b": 2, "c": 1}})["q"]
    assert diverse["coverage_20"] == redundant["coverage_20"] == 1
    assert diverse["alpha_ndcg_10"] > redundant["alpha_ndcg_10"]
    assert native_metrics(nuggets, {"q": {"a": 1}})["q"]["coverage_20"] == 0.5
