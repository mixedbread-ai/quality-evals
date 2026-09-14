import csv
import json
from statistics import fmean, median

import pytest

from evals_repro.benchmarks import Benchmark
from evals_repro.overview import overview
from evals_repro.protocol import timing_counts
from evals_repro.records import append_record, save_json
from evals_repro.study import BENCHMARKS


@pytest.fixture
def suite(tmp_path):
    benchmarks = []
    for index, name in enumerate(BENCHMARKS):
        subsets = ("a", "b") if name in ("text", "cure") else ("a",)
        benchmark = Benchmark(name, subsets, (), lambda name: None)
        benchmarks.append(benchmark)
        root = tmp_path / name / "monolingual"
        counts = timing_counts(name, list(subsets), 2, 42)
        stage = "mixedbread-markdown" if name == "vidore-v3" else "bm25"
        save_json(
            root / "manifest.json",
            {
                "config": {
                    "subsets": list(subsets),
                    "first_stage": stage,
                    "depth": 100,
                    "seed": 42,
                    "samples": 2,
                    "repeats": 5,
                    "implementation": {},
                    "timing_queries": counts,
                },
                "datasets": {s: {"queries": 2} for s in subsets},
                "models": {"mock": {"model": "mock"}},
                "selections": {s: [f"q{i}" for i in range(counts[s])] for s in subsets},
                "latency_host": {"hostname": "test", "location": "test"},
            },
        )
        for subset in subsets:
            for model in (stage, stage + "+mock@100"):
                value = (index + 1) / 10 if "+" in model else 0.1
                save_json(
                    root / "quality" / model / f"{subset}.json",
                    {
                        "method": model,
                        "subset": subset,
                        "native_language": "english",
                        "languages": [
                            {
                                "language": "english",
                                "num_queries": 2,
                                "metrics": {"ndcg_cut_10": value, "alpha_ndcg_10": value / 2},
                                "per_query": {"q0": {}, "q1": {}},
                            }
                        ],
                    },
                )
            for q in range(counts[subset]):
                for repeat in range(5):
                    append_record(
                        root / "latency.jsonl",
                        {
                            "subset": subset,
                            "query_id": f"q{q}",
                            "model": "mock",
                            "repeat": repeat,
                            "document_count": 100,
                            "latency_ns": (index + q + repeat + 1) * 1e6,
                            "wall_ns": (index + q + repeat + 1) * 1e6,
                            "attempts": [{}],
                        },
                    )
    return tmp_path, benchmarks


def test_overview_weights_benchmark_groups_and_exports_baselines(suite):
    root, benchmarks = suite
    rendered = overview(root, benchmarks)
    with (root / "quality.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    means = [r for r in rows if r["aggregation"] == "benchmark_mean"]
    average = next(r for r in rows if r["aggregation"] == "suite_mean")
    assert float(average["mock"]) == pytest.approx(fmean([10, 20, 30, 40, 25, 60, 70]), abs=0.005)
    assert len(means) == 7 and all(r["baseline"] for r in rows)
    vidore = [r for r in rows if r["benchmark"] == "vidore-v3"]
    assert all(r["first_stage"] == "mixedbread-markdown" for r in vidore)
    assert "| overall |" in rendered
    summary = json.loads((root / "latency-summary.json").read_text())
    saved = [json.loads(line) for p in root.glob("*/monolingual/latency.jsonl") for line in p.read_text().splitlines()]
    assert summary["overall"]["mock"]["complete"]
    assert summary["overall"]["mock"]["p50"]["ms"] == median(r["latency_ns"] / 1e6 for r in saved)


def test_overview_leaves_incomplete_quality_and_timing_averages_blank(suite):
    root, benchmarks = suite
    (root / "text/monolingual/quality/bm25+mock@100/b.json").unlink()
    path = root / "text/monolingual/latency.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    overview(root, benchmarks)
    with (root / "quality.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert next(r for r in rows if r["aggregation"] == "suite_mean")["mock"] == ""
    summary = json.loads((root / "latency-summary.json").read_text())
    assert not summary["overall"]["mock"]["complete"]


def test_overview_rejects_changed_queries(suite):
    root, benchmarks = suite
    path = root / "text/monolingual/quality/bm25+mock@100/a.json"
    result = json.loads(path.read_text())
    result["languages"][0]["per_query"] = {"q0": {}, "different": {}}
    save_json(path, result)
    with pytest.raises(ValueError, match="query coverage"):
        overview(root, benchmarks)


@pytest.mark.parametrize("change", ["host", "protocol", "model"])
def test_overview_rejects_incompatible_runs(suite, change):
    root, benchmarks = suite
    path = root / "cure/monolingual/manifest.json"
    manifest = json.loads(path.read_text())
    if change == "host":
        manifest["latency_host"]["hostname"] = "another"
    elif change == "protocol":
        manifest["config"]["seed"] = 1
    else:
        manifest["models"]["mock"]["model"] = "another"
    save_json(path, manifest)
    with pytest.raises(ValueError, match="differs"):
        overview(root, benchmarks)
