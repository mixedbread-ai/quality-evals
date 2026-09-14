from functools import partial

import pyndeval

from evals_repro.benchmarks import Benchmark
from evals_repro.benchmarks.hub import rows
from evals_repro.data import Query, Subset, check_consistency
from evals_repro.metrics import evaluate

QUERIES = ("freshstack/queries-oct-2024", "023ac3a14caf9d6ebb13d01adcfb8fa05e5a9630")
CORPUS = ("freshstack/corpus-oct-2024", "069f66dc323e163b48b10d08408d282733d4393b")
TOPICS = ("langchain", "yolo", "laravel", "angular", "godot")


def native_metrics(nuggets: dict, run: dict) -> dict:
    qrels = [
        pyndeval.SubtopicQrel(qid, nid, cid, grade)
        for qid, subtopics in nuggets.items()
        for nid, rels in subtopics.items()
        for cid, grade in rels.items()
    ]
    evaluator = pyndeval.RelevanceEvaluator(qrels, measures=["alpha-nDCG@10"])
    scores = {}
    for qid, subtopics in nuggets.items():
        hits = sorted(run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True)
        alpha = evaluator.evaluate([pyndeval.ScoredDoc(qid, cid, score) for cid, score in hits[:10]])
        coverage = evaluate(subtopics, dict.fromkeys(subtopics, run[qid]), frozenset({"success_20"}))
        scores[qid] = {
            "alpha_ndcg_10": alpha[qid]["alpha-nDCG@10"],
            "coverage_20": sum(v["success_20"] for v in coverage.values()) / len(subtopics),
        }
    return scores


def load_subset(name: str) -> Subset:
    pages = {str(d["_id"]): d["text"] for d in rows(CORPUS, f"{name}/train-00000-of-00001.parquet")}
    queries, qrels, nuggets = [], {}, {}
    for q in rows(QUERIES, f"{name}/test-00000-of-00001.parquet"):
        qid = str(q["query_id"])
        queries.append(Query(qid, q["query_title"] + " " + q["query_text"], "english"))
        qrels[qid], nuggets[qid] = {}, {}
        for nugget in q["nuggets"]:
            rels = dict.fromkeys(nugget["non_relevant_corpus_ids"], 0)
            rels.update(dict.fromkeys(nugget["relevant_corpus_ids"], 1))
            nuggets[qid][str(nugget["_id"])] = rels
            for cid, grade in rels.items():
                qrels[qid][cid] = qrels[qid].get(cid, 0) + grade
    subset = Subset(
        name,
        f"{QUERIES[0]}@{QUERIES[1]}+{CORPUS[0]}@{CORPUS[1]}/{name}",
        "english",
        pages,
        queries,
        qrels,
        annotations=nuggets,
        additional_metrics=partial(native_metrics, nuggets),
    )
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("freshstack", TOPICS, (), load_subset)
