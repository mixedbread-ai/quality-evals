from types import SimpleNamespace

import pytest

from evals_repro.latency import MeasuredReranker, repeat_effect, retry_after, statistics
from evals_repro.throttle import Budget


def repeated_rows(ratios):
    return [
        {
            "subset": str(q % 2),
            "query_id": str(q),
            "repeat": repeat,
            "latency_ns": (q + 1) * 1e6 * (ratio if repeat else 1),
            "attempts": [{}],
        }
        for q, ratio in enumerate(ratios)
        for repeat in range(5)
    ]


@pytest.mark.parametrize("ratio, status", [(0.25, "speedup"), (1, "no_material_speedup"), (1.4, "no_material_speedup")])
def test_repeat_check_detects_speedup_and_stable_or_slower_repeats(ratio, status):
    rows = repeated_rows([ratio] * 20)
    result = repeat_effect(rows)
    assert result["status"] == status
    assert result["paired_queries"] == 20 and result["excluded_queries"] == 0
    assert result["ratio"]["estimate"] == pytest.approx(ratio)
    assert result["ratio"]["ci95"] == pytest.approx([ratio, ratio])
    assert result == repeat_effect(list(reversed(rows)))


def test_repeat_check_does_not_claim_an_uncertain_speedup():
    result = repeat_effect(repeated_rows([0.5] * 10 + [1.2] * 10))
    assert result["status"] == "inconclusive"
    assert result["ratio"]["ci95"] == [0.5, 1.2]


def test_repeat_check_pairs_queries_despite_unequal_numbers_of_repeats():
    rows = [r for r in repeated_rows([1] * 20) if int(r["query_id"]) < 10 or r["repeat"] < 2]
    result = repeat_effect(rows)
    assert result["ratio"]["estimate"] == 1
    assert result["status"] == "no_material_speedup"


def test_repeat_check_excludes_retries_warmups_and_queries_without_first_calls():
    rows = repeated_rows([1] * 20)
    for row in rows:
        if row["repeat"] == 2 or row["query_id"] == "0":
            row.update(latency_ns=1, attempts=[{}, {}])
        if row["query_id"] == "1" and row["repeat"] == 0:
            row["repeat"] = -1
    result = repeat_effect(rows)
    assert result["paired_queries"] == 18 and result["excluded_queries"] == 2
    assert result["ratio"]["estimate"] == 1


@pytest.mark.parametrize("ratios", [[], [0.25] * 4])
def test_repeat_check_requires_enough_query_pairs(ratios):
    result = repeat_effect(repeated_rows(ratios))
    assert result["status"] == "insufficient_queries"
    assert "ratio" not in result


def test_repeat_check_rejects_duplicates():
    rows = repeated_rows([1] * 20)
    with pytest.raises(ValueError, match="Duplicate"):
        repeat_effect([*rows, rows[0]])


def test_retry_waits_are_excluded_from_successful_latency(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("evals_repro.latency.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("evals_repro.latency.time.perf_counter_ns", lambda: int(clock[0] * 1e9))
    monkeypatch.setattr("evals_repro.latency.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Limited(Exception):
        status_code = 429
        headers = {"retry-after": "10"}

    class Rerank:
        name = "test"
        calls = 0

        def rerank(self, query, documents):
            self.calls += 1
            clock[0] += 1
            if self.calls == 1:
                raise Limited()
            return [0.5]

    scores, timing = MeasuredReranker(Rerank()).run("q", ["d"])
    assert scores == [0.5]
    assert timing["latency_ns"] == 1_000_000_000
    assert timing["wall_ns"] == 12_000_000_000
    assert [a["status"] for a in timing["attempts"]] == [429, 200]


def test_authentication_errors_are_not_retried():
    error = Exception()
    error.status_code = 401
    assert retry_after(error, 0) is None


def test_token_budget_wait_is_outside_request_latency(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("evals_repro.latency.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("evals_repro.latency.time.perf_counter_ns", lambda: round(clock[0] * 1e9))
    monkeypatch.setattr("evals_repro.latency.time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def rerank(query, docs):
        clock[0] += 1
        return [0.5]

    runner = MeasuredReranker(SimpleNamespace(name="test", rerank=rerank), token_budget=Budget(1))
    monkeypatch.setattr(runner.token_budget.ready, "wait", lambda timeout: clock.__setitem__(0, clock[0] + timeout))
    runner.run("q", ["d"])
    _, timing = runner.run("q", ["d"])
    assert timing["latency_ns"] == 1_000_000_000
    assert timing["wall_ns"] >= 60_000_000_000
    assert len(timing["attempts"]) == 1


def test_incomplete_rankings_fail():
    model = SimpleNamespace(name="test", rerank=lambda q, d: [float("-inf")])
    with pytest.raises(ValueError, match="nonfinite"):
        MeasuredReranker(model).run("q", ["d"])


def test_cluster_bootstrap_retains_repeats():
    rows = [
        {"subset": "s", "query_id": q, "latency_ns": ms * 1e6, "attempts": [{}]}
        for q, ms in [("a", 1), ("a", 2), ("a", 3), ("b", 4), ("b", 5), ("b", 6)]
    ]
    result = statistics(rows, bootstraps=100)
    assert result["requests"] == 6 and result["queries"] == 2
    assert result["mean"] == {"ms": 3.5, "ci95_ms": [2.0, 5.0]}
    assert result["p90"]["ms"] == 5.5
    assert result == statistics(rows, bootstraps=100)


def test_http_status_retries_honor_retry_after():
    import httpx

    response = httpx.Response(429, headers={"Retry-After": "7"}, request=httpx.Request("POST", "https://test/rerank"))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    assert retry_after(caught.value, 0) == 7
