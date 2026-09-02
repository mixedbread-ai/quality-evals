from statistics import fmean

import pytrec_eval

from evals_repro.index import Run

Qrels = dict[str, dict[str, int]]
MEASURES = frozenset({"ndcg_cut_10", "ndcg_cut_5", "recall_10", "recall_50", "recall_100", "map_cut_10"})


def evaluate(qrels: Qrels, run: Run, measures: frozenset[str] = MEASURES) -> dict[str, dict[str, float]]:
    scored = pytrec_eval.RelevanceEvaluator(qrels, set(measures)).evaluate(run)
    empty = dict.fromkeys(measures, 0.0)
    return {qid: scored.get(qid, empty) for qid in qrels}


def summarize(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    measures = {m for scores in per_query.values() for m in scores}
    return {m: fmean(scores[m] for scores in per_query.values()) for m in sorted(measures)}
