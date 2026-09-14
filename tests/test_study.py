import json
from types import SimpleNamespace

import pytest

from evals_repro.data import Query, Subset
from evals_repro.study import candidates, read_records


def test_candidates_freeze_and_reject_changed_corpus(tmp_path):
    from evals_repro.bm25 import BM25Retriever

    query = Query("q", "alpha", "english")
    subset = Subset("test", "source", "english", {"a": "alpha", "b": "beta"}, [query], {})
    first = candidates(subset, [query], BM25Retriever(include_zero=True), tmp_path, 2)
    assert list(first["q"]["scores"]) == ["a", "b"]
    assert candidates(subset, [query], NoneRetriever(), tmp_path, 2) == first
    subset.pages["a"] = "changed"
    with pytest.raises(ValueError, match="Stale candidates"):
        candidates(subset, [query], NoneRetriever(), tmp_path, 2)


class NoneRetriever:
    name = "bm25"

    def run(self, *args):
        raise AssertionError("Cached candidates should not retrieve")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from evals_repro import study
    from evals_repro.benchmarks import Benchmark

    queries = [Query(str(i), f"alpha {i}", "english") for i in range(24)]
    subset = Subset(
        "test", "source", "english", {"a": "alpha", "b": "beta"}, queries, {q.id: {"b": 1} for q in queries}
    )
    benchmark = Benchmark("test", ("test",), (), lambda name: subset)
    args = SimpleNamespace(
        study_dir=tmp_path,
        subsets=None,
        depth=2,
        seed=42,
        samples=20,
        repeats=5,
        models=["mock"],
        store_prefix="test",
        voyage_interval=0,
        host_location="test",
        phase="quality",
        first_stage="bm25",
        cache_dir=tmp_path / "cache",
        quality_workers=2,
        voyage_tpm=2_000_000,
    )
    calls = []

    class Model:
        name = model = "mock"

        def rerank(self, query, documents):
            calls.append((query, documents))
            return [0.0, 1.0]

    monkeypatch.setitem(study.RERANKERS, "mock", Model)
    return args, benchmark, subset, calls


@pytest.mark.parametrize("repeats", [5, 10])
def test_quality_and_timing_resume_with_frozen_queries(experiment, repeats):
    from evals_repro.study import study

    args, benchmark, subset, calls = experiment
    args.repeats = repeats
    study(args, benchmark)
    assert len(calls) == 24
    args.phase = "latency"
    study(args, benchmark)
    assert len(calls) == 25 + 20 * repeats
    study(args, benchmark)
    assert len(calls) == 25 + 20 * repeats
    root = args.study_dir / "test" / "monolingual"
    assert len(read_records(root / "latency.jsonl")) == 20 * repeats
    assert len(read_records(root / "warmup.jsonl")) == 1
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(set(manifest["selections"]["test"])) == 20
    args.phase = "quality"
    study(args, benchmark)
    result = json.loads((root / "quality" / "bm25+mock@2" / "test.json").read_text())
    assert result["languages"][0]["num_queries"] == 24
    assert result["languages"][0]["metrics"]["ndcg_cut_10"] == 1
    assert len(calls) == 25 + 20 * repeats
    subset.pages["a"] = "changed"
    with pytest.raises(ValueError, match="dataset changed"):
        study(args, benchmark)


def test_preparation_needs_no_reranker_and_new_models_reuse_candidates(experiment, monkeypatch):
    from evals_repro import study

    args, benchmark, subset, calls = experiment
    args.phase = "prepare"
    args.models = ["unavailable"]
    study.study(args, benchmark)
    assert not calls
    args.phase, args.models = "quality", ["mock"]
    study.study(args, benchmark)

    class Second:
        model = name = "second"

        def rerank(self, query, docs):
            return [1.0, 0.0]

    monkeypatch.setitem(study.RERANKERS, "second", Second)
    args.models = ["second"]
    study.study(args, benchmark)
    root = args.study_dir / "test" / "monolingual"
    assert len(read_records(root / "quality.jsonl")) == 48
    assert len(read_records(root / "candidates/test.jsonl")) == 24


def test_markdown_store_timing_reuses_candidates(experiment, monkeypatch):
    from evals_repro import study
    from evals_repro.stores import StoreRetriever

    args, benchmark, _, calls = experiment
    args.phase, args.first_stage = "latency", "mixedbread-markdown"
    searches = []
    monkeypatch.setattr("evals_repro.clients.Mixedbread", lambda **kwargs: None)

    def search(self, store, query, depth):
        searches.append((store, self.content, self.search_options(), depth))
        return {"a": 1.0, "b": 0.0}

    monkeypatch.setattr(StoreRetriever, "search", search)
    study.study(args, benchmark)
    study.study(args, benchmark)
    assert searches == [("test-test-markdown", "markdown", {"rerank": False}, 2)] * 20
    assert len(calls) == 101


def test_sampling_is_order_independent_and_seeded():
    from evals_repro.study import select_queries

    queries = [Query(str(i), "text", "english") for i in range(100)]
    first = select_queries(queries, "dataset", 20, 42)
    assert len(first) == 20 and len({q.id for q in first}) == 20
    assert first == select_queries(list(reversed(queries)), "dataset", 20, 42)
    assert first != select_queries(queries, "dataset", 20, 43)


def test_timing_query_budget_is_per_dataset():
    from evals_repro.benchmarks import registry
    from evals_repro.protocol import timing_counts
    from evals_repro.study import BENCHMARKS

    benchmarks = registry()
    total = 0
    for name in BENCHMARKS:
        subsets = list(benchmarks[name].subsets)
        counts = timing_counts(name, subsets, 20, 42)
        assert counts == timing_counts(name, list(reversed(subsets)), 20, 42)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert sum(counts.values()) == (140 if name == "text" else 20)
        total += sum(counts.values())
    assert total * 5 * 4 == 5200


def test_family_timing_budget_and_report_coverage(experiment):
    from dataclasses import replace

    from evals_repro.study import study

    args, benchmark, subset, calls = experiment
    benchmark = replace(benchmark, subsets=("a", "b", "c"), load=lambda name: replace(subset, name=name))
    args.phase = "latency"
    study(args, benchmark)
    assert len(calls) == 103
    study(args, benchmark)
    assert len(calls) == 103
    root = args.study_dir / "test/monolingual"
    manifest = json.loads((root / "manifest.json").read_text())
    assert sorted(map(len, manifest["selections"].values())) == [6, 7, 7]
    summary = json.loads((root / "latency-summary.json").read_text())["overall"]["mock"]
    assert (summary["queries"], summary["requests"]) == (20, 100)
    assert "| overall | mock | 100/100 |" in (root / "report.md").read_text()
    args.phase = "quality"
    study(args, benchmark)
    assert len(read_records(root / "quality.jsonl")) == 72


def test_small_timing_budget_skips_unselected_subsets(experiment):
    from dataclasses import replace

    from evals_repro.study import study

    args, benchmark, subset, calls = experiment
    benchmark = replace(benchmark, subsets=("a", "b", "c"), load=lambda name: replace(subset, name=name))
    args.phase, args.samples = "latency", 1
    study(args, benchmark)
    assert len(calls) == 6


def test_truncated_record_recovery_and_corruption(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"ok":1}\n{"partial":')
    assert read_records(path) == [{"ok": 1}]
    assert path.read_bytes() == b'{"ok":1}\n'
    path.write_text('{"ok":1}')
    assert read_records(path) == [{"ok": 1}]
    assert path.read_bytes().endswith(b"\n")
    path.write_text('broken\n{"ok":1}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_records(path)


def test_resume_rejects_changed_candidates_and_timing_host(experiment):
    from evals_repro.study import study

    args, benchmark, _, _ = experiment
    args.phase = "latency"
    study(args, benchmark)
    args.host_location = "another site"
    with pytest.raises(ValueError, match="host changed"):
        study(args, benchmark)
    args.host_location = "test"
    path = args.study_dir / "test/monolingual/candidates/test.jsonl"
    rows = read_records(path)
    rows[0]["scores"]["a"] += 0.1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Stale candidates"):
        study(args, benchmark)


def test_timing_is_serial_and_wall_statistics_include_retries(experiment, monkeypatch):
    from evals_repro import study

    args, benchmark, _, _ = experiment
    args.phase = "latency"

    def forbidden(*args, **kwargs):
        raise AssertionError("Timing cannot use concurrent quality workers")

    monkeypatch.setattr(study, "ThreadPoolExecutor", forbidden)
    study.study(args, benchmark)
    rows = [{"subset": "s", "query_id": "q", "repeat": 0, "latency_ns": 1e6, "wall_ns": 7e6, "attempts": [{}, {}]}]
    summary = study.timing_summary(rows, 42)
    assert summary["mean"]["ms"] == 1
    assert summary["wall"]["mean"]["ms"] == 7
    assert summary["retried_requests"] == 1


def test_timing_resumes_after_failure_without_repeating_completed_requests(experiment, monkeypatch):
    from evals_repro import study

    args, benchmark, _, _ = experiment
    args.phase = "latency"

    class Failing:
        name = model = "mock"
        calls = 0
        fail = True

        def rerank(self, query, documents):
            Failing.calls += 1
            if Failing.fail and Failing.calls == 5:
                raise ValueError("interrupted")
            return [0.0, 1.0]

    monkeypatch.setitem(study.RERANKERS, "mock", Failing)
    with pytest.raises(ValueError, match="interrupted"):
        study.study(args, benchmark)
    path = args.study_dir / "test/monolingual/latency.jsonl"
    before = read_records(path)
    assert len(before) == 3
    Failing.fail = False
    study.study(args, benchmark)
    after = read_records(path)
    assert len(after) == 100 and after[:3] == before
    assert len({(r["query_id"], r["repeat"]) for r in after}) == 100


def test_rate_limited_provider_cannot_block_other_quality_providers(experiment, monkeypatch):
    from threading import Event

    from evals_repro import study

    args, benchmark, _, _ = experiment
    finished = Event()
    calls = []

    class Slow:
        name = model = provider = "slow"

        def rerank(self, query, documents):
            assert finished.wait(timeout=5), "Other provider was blocked by a rate-limit wait"
            return [0.0, 1.0]

    class Fast:
        name = model = provider = "fast"

        def rerank(self, query, documents):
            calls.append(query)
            if len(calls) == 24:
                finished.set()
            return [0.0, 1.0]

    monkeypatch.setitem(study.RERANKERS, "slow", Slow)
    monkeypatch.setitem(study.RERANKERS, "fast", Fast)
    args.models, args.quality_workers = ["slow", "fast"], 1
    study.study(args, benchmark)
    rows = read_records(args.study_dir / "test/monolingual/quality.jsonl")
    assert len(rows) == 48 and len(calls) == 24


def test_latency_server_checks_are_saved_and_must_match_on_resume(experiment, monkeypatch):
    from evals_repro import study

    args, benchmark, _, calls = experiment
    args.phase = "latency"
    checks = {"http://local": {"prefix_caching": False}}
    monkeypatch.setattr(study.RERANKERS["mock"], "validate_latency", lambda self: checks, raising=False)
    study.study(args, benchmark)
    manifest = json.loads((args.study_dir / "test/monolingual/manifest.json").read_text())
    assert manifest["sessions"][-1]["latency_checks"] == {"mock": checks}
    checks.clear()
    with pytest.raises(ValueError, match="latency server checks changed"):
        study.study(args, benchmark)
    assert len(calls) == 101


def test_latency_preflight_failure_prevents_measured_calls(experiment, monkeypatch):
    from evals_repro import study

    args, benchmark, _, calls = experiment
    args.phase = "latency"

    def reject(self):
        raise ValueError("Prefix caching enabled")

    monkeypatch.setattr(study.RERANKERS["mock"], "validate_latency", reject, raising=False)
    with pytest.raises(ValueError, match="Prefix caching enabled"):
        study.study(args, benchmark)
    assert not calls


def test_report_rebuilds_summaries_without_clients_or_datasets(experiment, monkeypatch):
    from dataclasses import replace

    from evals_repro import study

    args, benchmark, _, calls = experiment
    args.phase = "latency"
    study.study(args, benchmark)
    root = args.study_dir / "test/monolingual"
    (root / "latency-summary.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must use saved artifacts only")

    monkeypatch.setitem(study.RERANKERS, "mock", forbidden)
    args.phase = "report"
    study.study(args, replace(benchmark, load=forbidden))
    summary = json.loads((root / "latency-summary.json").read_text())["overall"]["mock"]
    assert summary["requests"] == 100
    assert summary["repeat_effect"]["paired_queries"] == 20
    assert len(calls) == 101
