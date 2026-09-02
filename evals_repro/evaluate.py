import json
from dataclasses import asdict, dataclass
from pathlib import Path

from evals_repro.data import Subset
from evals_repro.metrics import evaluate, summarize
from evals_repro.retrievers import Retriever


@dataclass(frozen=True)
class LanguageResult:
    language: str
    num_queries: int
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SubsetResult:
    method: str
    subset: str
    native_language: str
    top_k: int
    languages: list[LanguageResult]

    def save(self, results_dir: Path) -> Path:
        path = result_path(results_dir, self.method, self.subset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path


def result_path(results_dir: Path, method: str, subset: str) -> Path:
    return results_dir / method / f"{subset}.json"


def evaluate_subset(retriever: Retriever, subset: Subset, languages: list[str], top_k: int) -> SubsetResult:
    results = []
    for language in languages:
        queries = subset.queries_in(language)
        per_query = evaluate(subset.qrels_for(queries), retriever.run(subset, queries, top_k))
        results.append(LanguageResult(language, len(per_query), summarize(per_query), per_query))
    return SubsetResult(retriever.name, subset.name, subset.native_language, top_k, results)
