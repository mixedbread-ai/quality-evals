import csv
import json
import math
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean

from filelock import FileLock

from evals_repro.benchmarks import Benchmark
from evals_repro.protocol import timing_counts
from evals_repro.records import read_records, save_json
from evals_repro.study import write_summary


def quality_rows(root: Path, benchmark: Benchmark, manifest: dict) -> list[dict]:
    config = manifest["config"]
    metric = "alpha_ndcg_10" if benchmark.name == "freshstack" else "ndcg_cut_10"
    methods = {f"{config['first_stage']}+{m}@{config['depth']}": m for m in manifest["models"]}
    methods[config["first_stage"]] = "baseline"
    rows, queries = {}, {}
    for path in sorted((root / "quality").glob("*/*.json")):
        result = json.loads(path.read_text())
        subset, method = result["subset"], result["method"]
        if subset not in config["subsets"]:
            raise ValueError(f"{path}: unexpected subset")
        if method not in methods:
            raise ValueError(f"{path}: unexpected model or first stage")
        model = methods[method]
        native = next(r for r in result["languages"] if r["language"] == result["native_language"])
        ids = set(native["per_query"])
        expected = manifest["datasets"][subset]["queries"] // max(1, len(benchmark.languages))
        if len(ids) != native["num_queries"] or len(ids) != expected or queries.setdefault(subset, ids) != ids:
            raise ValueError(f"{path}: incomplete or mismatched query coverage")
        row = rows.setdefault(
            subset,
            {
                "benchmark": benchmark.name,
                "dataset": subset,
                "aggregation": "subset",
                "queries": len(ids),
                "metric": metric,
                "first_stage": config["first_stage"],
            },
        )
        value = 100 * native["metrics"][metric]
        if model in row or not math.isfinite(value):
            raise ValueError(f"{path}: duplicate or nonfinite result")
        row[model] = value
    return [rows[s] for s in config["subsets"] if s in rows]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in row.items()} for row in rows)


def collect_timings(studies) -> tuple[list[dict], dict | None, dict]:
    timings, expected = [], {}
    timing_protocol = None
    for benchmark, root, manifest in studies:
        config = manifest["config"]
        records = read_records(root / "latency.jsonl")
        if not records:
            continue
        protocol = {k: config[k] for k in ("seed", "samples", "repeats", "depth", "implementation")}
        protocol["host"] = manifest["latency_host"]
        if timing_protocol is not None and timing_protocol != protocol:
            raise ValueError("Timing host or protocol differs across benchmarks")
        timing_protocol = protocol
        counts = timing_counts(benchmark.name, config["subsets"], config["samples"], config["seed"])
        if config["timing_queries"] != counts:
            raise ValueError(f"{root}: timing query allocation differs from the protocol")
        keys = set()
        for row in records:
            subset, qid, model, repeat = (row[k] for k in ("subset", "query_id", "model", "repeat"))
            key = (subset, qid, model, repeat)
            if (
                key in keys
                or qid not in manifest["selections"].get(subset, [])
                or model not in manifest["models"]
                or repeat not in range(config["repeats"])
                or row["document_count"] != config["depth"]
            ):
                raise ValueError(f"{root}: duplicate or unexpected latency record")
            keys.add(key)
            timings.append({**row, "subset": f"{benchmark.name}/{subset}"})
        for model in manifest["models"]:
            complete = set(config["subsets"]) == set(benchmark.subsets) and all(
                len(set(manifest["selections"].get(s, []))) == n for s, n in config["timing_queries"].items()
            )
            if complete:
                expected.setdefault(model, {})[benchmark.name] = (
                    sum(config["timing_queries"].values()) * config["repeats"]
                )
    return timings, timing_protocol, expected


def overview(directory: Path, benchmarks: list[Benchmark]) -> str:
    with ExitStack() as stack:
        studies = []
        for benchmark in benchmarks:
            root = directory / benchmark.name / "monolingual"
            if not (root / "manifest.json").exists():
                continue
            stack.enter_context(FileLock(str(root / ".lock"), timeout=0))
            studies.append((benchmark, root, json.loads((root / "manifest.json").read_text())))
        if not studies:
            raise ValueError("No study manifests found")
        models, metadata = [], {}
        rows, means = [], []
        for benchmark, root, manifest in studies:
            for model, spec in manifest["models"].items():
                if metadata.setdefault(model, spec) != spec:
                    raise ValueError(f"{model}: model configuration differs across benchmarks")
                if model not in models:
                    models.append(model)
            subset_rows = quality_rows(root, benchmark, manifest)
            rows.extend(subset_rows)
            config = manifest["config"]
            mean = {
                "benchmark": benchmark.name,
                "dataset": "macro_average",
                "aggregation": "benchmark_mean",
                "queries": sum(row["queries"] for row in subset_rows),
                "metric": "alpha_ndcg_10" if benchmark.name == "freshstack" else "ndcg_cut_10",
                "first_stage": config["first_stage"],
            }
            for model in ["baseline", *models]:
                values = [row[model] for row in subset_rows if model in row]
                if len(values) == len(benchmark.subsets) and set(config["subsets"]) == set(benchmark.subsets):
                    mean[model] = fmean(values)
            means.append(mean)
        timings, timing_protocol, expected = collect_timings(studies)
        average = {
            "benchmark": "overall",
            "dataset": "seven_group_average",
            "aggregation": "suite_mean",
            "metric": "composite",
        }
        for model in ["baseline", *models]:
            values = [row[model] for row in means if model in row]
            if len(values) == len(benchmarks):
                average[model] = fmean(values)
        columns = ["baseline", *models]
        rendered = [
            "# Reranker overview",
            "",
            "Scores ×100: nDCG@10, except FreshStack uses alpha-nDCG@10. Each benchmark averages its subsets equally.",
            "The overall score weights the seven benchmark groups equally; incomplete averages are blank.",
            "",
            "| Benchmark | " + " | ".join(columns) + " |",
            "|---|" + "---:|" * len(columns),
        ]
        for row in [*means, average]:
            rendered.append(
                f"| {row['benchmark']} | " + " | ".join(f"{row[m]:.2f}" if m in row else "" for m in columns) + " |"
            )
        write_csv(
            directory / "quality.csv",
            [*rows, *means, average],
            ["benchmark", "dataset", "aggregation", "queries", "metric", "first_stage", *columns],
        )
        if timings:
            write_summary(directory, timings, timing_protocol["seed"])
            path = directory / "latency-summary.json"
            summary = json.loads(path.read_text())
            summary["protocol"] = timing_protocol
            rendered += [
                "",
                "Overall p50 pools measured requests across datasets; incomplete runs have no overall p50.",
                "Inspect repeat-effect diagnostics before interpreting pooled latencies.",
                "",
                "| Model | Requests | Expected | Overall p50 ms | Repeat check |",
                "|---|---:|---:|---:|---|",
            ]
            for model, stats in summary["overall"].items():
                counts = expected.get(model, {})
                total = sum(counts.values()) if len(counts) == len(benchmarks) else None
                stats["complete"] = total is not None and total == stats["requests"]
                stats["expected_requests"] = total
                p50 = f"{stats['p50']['ms']:.1f}" if stats["complete"] else ""
                rendered.append(
                    f"| {model} | {stats['requests']} | {total or ''} | {p50} | {stats['repeat_effect']['status']} |"
                )
            save_json(path, summary)
        result = "\n".join(rendered) + "\n"
        (directory / "report.md").write_text(result)
        return result
