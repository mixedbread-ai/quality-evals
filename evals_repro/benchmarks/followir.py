from functools import partial
from statistics import fmean

from evals_repro.benchmarks import Benchmark
from evals_repro.benchmarks.hub import rows
from evals_repro.data import Query, Subset, check_consistency

SOURCES = {
    "robust04": ("mteb/Robust04InstructionRetrieval", "0495a5cb69aa8fa5bca81bd56ac248911c19eeb6"),
    "core17": ("mteb/Core17InstructionRetrieval", "67d8733dd0691f08f34ed9dda5941578cba8d91c"),
    "news21": ("mteb/News21InstructionRetrieval", "c6fffeb9cdd95c1ffa81c62c687b48b56352872b"),
}


def paired_metrics(changed: dict, run: dict) -> dict:
    metrics = {}
    for qid, doc_ids in changed.items():
        original, modified = run[qid + "-og"], run[qid + "-changed"]
        if original.keys() != modified.keys():
            raise ValueError(f"{qid}: FollowIR instruction pairs require identical candidates")
        ranks = [
            {cid: i for i, (cid, _) in enumerate(sorted(r.items(), key=lambda x: (x[1], x[0]), reverse=True), 1)}
            for r in (original, modified)
        ]
        deltas = []
        for cid in doc_ids:
            before, after = (r.get(cid, len(r) + 1) for r in ranks)
            deltas.append(after / before - 1 if before >= after else 1 - before / after)
        result = {
            "p_mrr_top100": fmean(deltas),
            "changed_document_coverage": sum(cid in original for cid in doc_ids) / len(doc_ids),
        }

        metrics[qid + "-og"] = metrics[qid + "-changed"] = result
    return metrics


def load_subset(name: str) -> Subset:
    source = SOURCES[name]
    docs = rows(source, "corpus/test-00000-of-00001.parquet")
    pages = {str(d["id"]): "\n".join(filter(None, (d.get("title"), d["text"]))).strip() for d in docs}
    queries, retrieval_text = [], {}
    for q in rows(source, "queries/test-00000-of-00001.parquet"):
        qid = str(q["id"])
        queries.append(Query(qid, q["text"] + "\n\n" + q["instruction"], "english"))
        retrieval_text[qid] = q["text"]
    qrels = {}
    for r in rows(source, "qrels/test-00000-of-00001.parquet"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    changed = {str(r["query-id"]): r["corpus-ids"] for r in rows(source, "qrel_diff/qrel_diff-00000-of-00001.parquet")}
    subset = Subset(
        name,
        f"{source[0]}@{source[1]}",
        "english",
        pages,
        queries,
        qrels,
        retrieval_text=retrieval_text,
        annotations=changed,
        additional_metrics=partial(paired_metrics, changed),
    )
    check_consistency(subset)
    return subset


BENCHMARK = Benchmark("followir", tuple(SOURCES), (), load_subset)
