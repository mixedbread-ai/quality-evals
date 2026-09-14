from evals_repro.benchmarks import Benchmark
from evals_repro.benchmarks.hub import rows
from evals_repro.data import Query, Subset, check_consistency

SOURCE = ("embedding-benchmark/FinanceBench", "e68478442112cae36b70a216f52cc2777acf0a7e")


def load_subset(name: str) -> Subset:
    pages = {
        str(d["id"]): "\n".join(filter(None, (d.get("title"), d["text"]))).strip() for d in rows(SOURCE, "corpus.jsonl")
    }
    queries = [Query(str(q["id"]), q["text"], "english") for q in rows(SOURCE, "queries.jsonl")]
    qrels = {}
    for r in rows(SOURCE, "relevance.jsonl"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    subset = Subset(name, "@".join(SOURCE), "english", pages, queries, qrels)
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("financebench", ("financebench",), (), load_subset)
