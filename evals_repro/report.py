import json
from pathlib import Path
from statistics import fmean

from evals_repro.benchmarks import Benchmark

Scores = dict[str, dict[str, dict[str, float]]]


def load_scores(results_dir: Path, measure: str) -> tuple[Scores, dict[str, str]]:
    scores: Scores = {}
    native: dict[str, str] = {}
    for path in sorted(results_dir.glob("*/*.json")):
        result = json.loads(path.read_text())
        by_language = {lr["language"]: lr["metrics"][measure] for lr in result["languages"]}
        scores.setdefault(result["method"], {})[result["subset"]] = by_language
        native[result["subset"]] = result["native_language"]
    return scores, native


def monolingual(by_subset: dict[str, dict[str, float]], native: dict[str, str]) -> dict[str, float]:
    return {s: langs[native[s]] for s, langs in by_subset.items() if native[s] in langs}


def crosslingual(by_subset: dict[str, dict[str, float]], languages: tuple[str, ...]) -> dict[str, float]:
    complete = {s: v for s, v in by_subset.items() if set(languages) <= v.keys()}
    return {s: fmean(v[lang] for lang in languages) for s, v in complete.items()}


def table(title: str, rows: dict[str, dict[str, float]], subsets: tuple[str, ...]) -> str:
    columns = [s for s in subsets if any(s in r for r in rows.values())]
    header = f"| {title} | " + " | ".join(columns) + " | avg |"
    rule = "|" + "---|" * (len(columns) + 2)
    lines = [header, rule]
    for method, scores in rows.items():
        if not scores:
            continue
        cells = [f"{100 * scores[c]:.2f}" if c in scores else "" for c in columns]
        avg = f"{100 * fmean(scores.values()):.2f}" if len(scores) == len(subsets) else ""
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {avg} |")
    return "\n".join(lines)


def report(
    results_dir: Path, benchmark: Benchmark, measure: str = "ndcg_cut_10", first_stage_only: bool = False
) -> str:
    scores, native = load_scores(results_dir, measure)
    if first_stage_only:
        scores = {m: v for m, v in scores.items() if "+" not in m}
    mono = {m: monolingual(v, native) for m, v in scores.items()}
    cross = {m: crosslingual(v, benchmark.languages) for m, v in scores.items()}
    tables = [table(f"monolingual {measure}", mono, benchmark.subsets)]
    if benchmark.crosslingual:
        tables.append(table(f"crosslingual {measure}", cross, benchmark.subsets))
    return "\n\n".join(tables)


def rerank_report(root: Path, benchmark: Benchmark) -> str:
    manifest = json.loads((root / "manifest.json").read_text())
    subsets = tuple(manifest["config"]["subsets"])
    lines = [
        f"# {benchmark.name} reranker study",
        "",
        f"Monolingual, {manifest['config']['first_stage']} top-{manifest['config']['depth']} candidates; scores ×100.",
    ]
    measures = {"ndcg_cut_10": "nDCG@10"}
    if benchmark.name == "freshstack":
        measures = {"alpha_ndcg_10": "alpha-nDCG@10", "coverage_20": "Nugget coverage@20"}
    elif benchmark.name == "followir":
        measures.update(map="MAP", p_mrr_top100="p-MRR")
        lines += [
            "",
            "BM25 top-100 adaptation: instruction pairs share candidates; missing documents receive rank 101.",
            "MAP uses the full returned ranking. These scores depend on the depth-100 candidate pool.",
        ]
    for measure, label in measures.items():
        scores, native = load_scores(root / "quality", measure)
        if scores:
            lines += ["", table(label, {m: monolingual(v, native) for m, v in scores.items()}, subsets)]
    path = root / "latency-summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
        host = manifest["latency_host"]
        lines += [
            "",
            f"Timing host: {host['hostname']}; {host['location']}.",
            "",
            "Serial requests; excluded warmups are saved separately. Brackets contain 95% query-cluster intervals.",
            "Successful-call time includes network and parsing. Wall time also includes retries and rate-limit waits.",
            "",
            "| Subset | Model | Requests | Mean ms | p50 ms | p90 ms | Wall mean ms | Retried |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for subset, models in [("overall", summary["overall"]), *summary["subsets"].items()]:
            for model, stats in models.items():
                cells = []
                for estimate in [stats[m] for m in ("mean", "p50", "p90")] + [stats["wall"]["mean"]]:
                    low, high = estimate["ci95_ms"]
                    cells.append(f"{estimate['ms']:.1f} [{low:.1f}, {high:.1f}]")
                counts = manifest["config"]["timing_queries"]
                expected = (sum(counts.values()) if subset == "overall" else counts[subset]) * manifest["config"][
                    "repeats"
                ]
                lines.append(
                    f"| {subset} | {model} | {stats['requests']}/{expected} | "
                    + " | ".join(cells)
                    + f" | {stats['retried_requests']} |"
                )
        lines += [
            "",
            "Repeat check: each query's median later call / first measured call, excluding retried calls.",
            "A paired, subset-stratified bootstrap tests for a reduction exceeding 10%; requires five query pairs.",
            "Investigate speedup flags before pooling repetitions. No detected effect does not prove no caching.",
            "First calls may already be warm; this check cannot identify the cause of a speedup.",
            "",
            "| Model | Paired queries | First p50 ms | Repeat p50 ms | Repeat / first [95% CI] | Check |",
            "|---|---:|---:|---:|---:|---|",
        ]
        labels = {
            "speedup": "REPEAT SPEEDUP",
            "no_material_speedup": "No material speedup detected",
            "inconclusive": "Inconclusive",
            "insufficient_queries": "Insufficient query pairs",
        }
        for model, stats in summary["overall"].items():
            check = stats.get("repeat_effect", {"status": "insufficient_queries", "paired_queries": 0})
            cells = [f"{check[key]:.1f}" if key in check else "—" for key in ("first_p50_ms", "repeat_p50_ms")]
            ratio = check.get("ratio")
            cells.append(f"{ratio['estimate']:.2f} [{ratio['ci95'][0]:.2f}, {ratio['ci95'][1]:.2f}]" if ratio else "—")
            lines.append(
                f"| {model} | {check['paired_queries']} | " + " | ".join(cells) + f" | {labels[check['status']]} |"
            )
    return "\n".join(lines) + "\n"
