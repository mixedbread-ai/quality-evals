import math

from evals_repro.metrics import evaluate, summarize


def test_perfect_and_missing_queries():
    qrels = {"q1": {"a": 2, "b": 1}, "q2": {"c": 1}}
    run = {"q1": {"a": 3.0, "b": 2.0, "x": 1.0}}
    per_query = evaluate(qrels, run)
    assert per_query["q1"]["ndcg_cut_10"] == 1.0
    assert per_query["q2"]["ndcg_cut_10"] == 0.0
    assert summarize(per_query)["ndcg_cut_10"] == 0.5


def test_graded_ndcg_uses_linear_gain():
    qrels = {"q": {"a": 2, "b": 1}}
    run = {"q": {"b": 2.0, "a": 1.0}}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(evaluate(qrels, run)["q"]["ndcg_cut_10"], dcg / ideal)
