from evals_repro.benchmarks import Benchmark
from evals_repro.benchmarks.hub import rows
from evals_repro.data import Query, Subset, check_consistency

REPO = "isaacus/legal-rag-bench"
REVISION = "db0b31dc6d195ce9916897e1ac5e4e6209736c8a"


def load_subset(name: str) -> Subset:
    passages = rows((REPO, REVISION), "corpus.jsonl")
    questions = list(rows((REPO, REVISION), "qa.jsonl"))
    subset = Subset(
        name,
        f"{REPO}@{REVISION}",
        "english",
        {str(p["id"]): p["text"] for p in passages},
        [Query(str(q["id"]), q["question"], "english") for q in questions],
        {str(q["id"]): {str(q["relevant_passage_id"]): 1} for q in questions},
    )
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("legalragbench", ("legalragbench",), (), load_subset)
