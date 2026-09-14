from evals_repro.benchmarks import Benchmark
from evals_repro.benchmarks.hub import rows
from evals_repro.data import Query, Subset, check_consistency

SOURCE = ("clinia/CUREv1", "072343ca2353c3dc832cf18cf0fef7c9a5a207b7")
SPECIALTIES = (
    "dentistry_and_oral_health",
    "dermatology",
    "gastroenterology",
    "genetics",
    "neuroscience_and_neurology",
    "orthopedic_surgery",
    "otorhinolaryngology",
    "plastic_surgery",
    "psychiatry_and_psychology",
    "pulmonology",
)


def load_subset(name: str) -> Subset:
    pages = {
        str(d["_id"]): "\n".join(filter(None, (d.get("title"), d["text"]))).strip()
        for d in rows(SOURCE, f"{name}/corpus.jsonl")
    }
    queries = [Query(str(q["_id"]), q["text"], "english") for q in rows(SOURCE, f"{name}/queries-en.jsonl")]
    qrels = {}
    for r in rows(SOURCE, f"{name}/qrels.jsonl"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    subset = Subset(name, "@".join(SOURCE) + "/" + name, "english", pages, queries, qrels)
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("cure", SPECIALTIES, (), load_subset)
