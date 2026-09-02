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
