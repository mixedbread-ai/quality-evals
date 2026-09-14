import json
import math
import platform
import random
import socket
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock

from evals_repro.benchmarks import Benchmark
from evals_repro.bm25 import BM25Retriever
from evals_repro.candidates import candidates
from evals_repro.data import Query, Subset
from evals_repro.evaluate import LanguageResult, SubsetResult
from evals_repro.latency import MeasuredReranker, repeat_effect, statistics
from evals_repro.metrics import evaluate, summarize
from evals_repro.protocol import manifest_for, register_subset
from evals_repro.records import append_record, read_records, save_json
from evals_repro.report import rerank_report
from evals_repro.rerank import RERANKERS
from evals_repro.stores import StoreRetriever
from evals_repro.throttle import Budget

METHODS = ("mixedbread", "voyage", "cohere-pro", "cohere-fast")
BENCHMARKS = ("text", "financebench", "legalragbench", "cure", "freshstack", "followir", "vidore-v3")


def select_queries(queries: list[Query], name: str, count: int, seed: int) -> list[Query]:
    if count > len(queries):
        raise ValueError(f"{name}: requested {count} timing queries, only {len(queries)} available")
    return random.Random(f"{seed}:{name}").sample(sorted(queries, key=lambda q: q.id), count)


def measure(runner: MeasuredReranker, subset: Subset, query: Query, frozen: dict, repeat: int) -> dict:
    ids = list(frozen["scores"])
    documents = subset.documents(ids)
    scores, timing = runner.run(query.text, documents)
    context = getattr(runner.reranker, "timing_context", None)
    return {
        "subset": subset.name,
        "query_id": query.id,
        "language": query.language,
        "model": runner.reranker.name,
        "repeat": repeat,
        "candidate_digest": frozen["digest"],
        "document_count": len(ids),
        "document_bytes": sum(len(d.encode()) for d in documents),
        "scores": dict(zip(ids, scores, strict=True)),
        "truncated_documents": getattr(context, "truncated_documents", 0),
        **timing,
    }


def collect(args, subset: Subset, queries: list[Query], frozen: dict, root: Path, runners: dict, records: list) -> None:
    done = set()
    for row in records:
        if row["subset"] != subset.name:
            continue
        qid = row["query_id"]
        key = (qid, row["model"], row["repeat"])
        if key in done or qid not in frozen or row["candidate_digest"] != frozen[qid]["digest"]:
            raise ValueError(f"{subset.name}: duplicate or stale measurements")
        if row["scores"].keys() != frozen[qid]["scores"].keys() or not all(map(math.isfinite, row["scores"].values())):
            raise ValueError(f"{subset.name}/{qid}: incomplete saved ranking")
        done.add(key)
    jobs = []
    for repeat in range(args.repeats if args.phase == "latency" else 1):
        order = list(queries)
        random.Random(f"{args.seed}:{subset.name}:{repeat}").shuffle(order)
        for query in order:
            models = list(runners)
            random.Random(f"{args.seed}:{subset.name}:{repeat}:{query.id}").shuffle(models)
            jobs.extend((query, model, repeat) for model in models if (query.id, model, repeat) not in done)
    if not jobs:
        return
    if args.phase == "latency":
        for model in dict.fromkeys(model for _, model, _ in jobs):
            query = jobs[0][0]
            warmup = measure(runners[model], subset, query, frozen[query.id], -1)
            append_record(root / "warmup.jsonl", warmup)

    def invoke(job):
        query, model, repeat = job
        return measure(runners[model], subset, query, frozen[query.id], repeat)

    def record(row):
        append_record(root / f"{args.phase}.jsonl", row)
        records.append(row)

    if args.phase == "latency":
        for job in jobs:
            record(invoke(job))
    else:
        with ExitStack() as stack:
            providers = {name: getattr(r.reranker, "provider", name) for name, r in runners.items()}
            pools = {
                p: stack.enter_context(ThreadPoolExecutor(args.quality_workers))
                for p in sorted(set(providers.values()))
            }
            pending = [pools[providers[job[1]]].submit(invoke, job) for job in jobs]
            error = None
            try:
                for future in as_completed(pending):
                    try:
                        record(future.result())
                    except CancelledError:
                        pass
                    except Exception as failure:
                        error = error or failure
                        for other in pending:
                            other.cancel()
                if error:
                    raise error
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    print(f"{args.phase}: {subset.name}, {len(jobs)} requests complete", flush=True)


def score_quality(subset: Subset, queries: list[Query], frozen: dict, records: list[dict], root: Path, args) -> None:
    runs = {args.first_stage: {qid: row["scores"] for qid, row in frozen.items()}}
    for row in records:
        if row["subset"] == subset.name:
            runs.setdefault(f"{args.first_stage}+{row['model']}@{args.depth}", {})[row["query_id"]] = row["scores"]
    for model, run in runs.items():
        if set(run) != {q.id for q in queries}:
            continue
        per_query = evaluate(subset.qrels_for(queries), run)
        if subset.additional_metrics:
            for qid, metrics in subset.additional_metrics(run).items():
                per_query[qid].update(metrics)
        result = SubsetResult(
            model,
            subset.name,
            subset.native_language,
            args.depth,
            [LanguageResult(subset.native_language, len(queries), summarize(per_query), per_query)],
        )
        result.save(root / "quality")


def timing_summary(records: list[dict], seed: int) -> dict:
    return {
        **statistics(records, seed),
        "wall": statistics(records, seed, clock="wall_ns"),
        "repeat_effect": repeat_effect(records, seed),
    }


def write_summary(root: Path, records: list[dict], seed: int) -> None:
    models = sorted({row["model"] for row in records})
    subsets = sorted({row["subset"] for row in records})
    save_json(
        root / "latency-summary.json",
        {
            "interval": "95% percentile bootstrap; 2000 query-cluster resamples stratified by subset",
            "timing": "Complete successful reranker call; wall time additionally includes failed attempts and waits",
            "overall": {m: timing_summary([r for r in records if r["model"] == m], seed) for m in models},
            "subsets": {
                s: {
                    m: timing_summary([r for r in records if r["model"] == m and r["subset"] == s], seed)
                    for m in models
                    if any(r["model"] == m and r["subset"] == s for r in records)
                }
                for s in subsets
            },
        },
    )


def study(args, benchmark: Benchmark) -> None:
    root = args.study_dir / benchmark.name / "monolingual"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".lock"), timeout=0):
        if args.phase == "report":
            manifest = json.loads((root / "manifest.json").read_text())
            records = read_records(root / "latency.jsonl")
            if records:
                write_summary(root, records, manifest["config"]["seed"])
            rendered = rerank_report(root, benchmark)
            (root / "report.md").write_text(rendered)
            print(rendered)
            return
        execute(args, benchmark, root)


def execute(args, benchmark: Benchmark, root: Path) -> None:
    manifest = manifest_for(args, benchmark, root)
    names = manifest["config"]["subsets"]
    if not set(names) <= set(benchmark.subsets):
        raise ValueError("Unknown benchmark subset")
    if args.phase == "latency":
        host = {"hostname": socket.gethostname(), "location": args.host_location, "platform": platform.platform()}
        if manifest.setdefault("latency_host", host) != host:
            raise ValueError("Latency host changed; use a new --study-dir")
    budget = Budget(args.voyage_tpm)
    runners = {}
    latency_checks = {}
    for key in args.models if args.phase != "prepare" else ():
        reranker = RERANKERS[key]()
        if args.phase == "latency" and hasattr(reranker, "validate_latency"):
            check = latency_checks[reranker.name] = reranker.validate_latency()
            for session in manifest["sessions"]:
                if (
                    session["phase"] == "latency"
                    and reranker.name in session["models"]
                    and session.get("latency_checks", {}).get(reranker.name) != check
                ):
                    raise ValueError(f"{reranker.name}: latency server checks changed; use a new --study-dir")
        package = "voyageai" if key.startswith("voyage") else "cohere" if key.startswith("cohere") else key
        metadata = {"model": reranker.model, **getattr(reranker, "metadata", {})}
        if package in ("voyageai", "cohere", "mixedbread"):
            metadata["sdk_version"] = version(package)
        if hasattr(reranker, "max_tokens_per_doc"):
            metadata["max_tokens_per_doc"] = reranker.max_tokens_per_doc
        if manifest["models"].setdefault(reranker.name, metadata) != metadata:
            raise ValueError(f"{reranker.name}: model configuration changed")
        runners[reranker.name] = MeasuredReranker(
            reranker,
            args.voyage_interval if key.startswith("voyage") else 0,
            token_budget=budget if key.startswith("voyage") else None,
        )
    manifest["sessions"].append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "phase": args.phase,
            "models": list(runners),
            "latency_checks": latency_checks,
            "quality_workers": args.quality_workers,
            "voyage_tpm": args.voyage_tpm,
            "voyage_interval": args.voyage_interval,
            "python": platform.python_version(),
            "packages": {
                p: version(p)
                for p in (
                    "cohere",
                    "voyageai",
                    "mixedbread",
                    "numpy",
                    "pytrec-eval-terrier",
                    "bm25s",
                    "pystemmer",
                    "ir-datasets",
                    "pyndeval",
                    "tokenizers",
                )
            },
        }
    )
    save_json(root / "manifest.json", manifest)
    retriever = (
        BM25Retriever(cache_dir=args.cache_dir, include_zero=True)
        if args.first_stage == "bm25"
        else StoreRetriever(args.store_prefix, "markdown")
    )
    records = read_records(root / f"{args.phase}.jsonl")
    for name in names:
        count = manifest["config"]["timing_queries"][name]
        if args.phase == "latency" and count == 0:
            continue
        subset = benchmark.load(name)
        register_subset(manifest, subset)
        queries = subset.queries_in(subset.native_language)
        selection = [q.id for q in select_queries(queries, name, count, args.seed)]
        if manifest["selections"].setdefault(name, selection) != selection:
            raise ValueError(f"{name}: timing query selection changed")
        save_json(root / "manifest.json", manifest)
        selected = [q for q in queries if q.id in selection] if args.phase == "latency" else queries
        frozen = candidates(subset, selected, retriever, root, args.depth)
        if args.phase != "prepare":
            collect(args, subset, selected, frozen, root, runners, records)
        if args.phase == "latency":
            write_summary(root, records, args.seed)
        else:
            score_quality(subset, queries, frozen, records, root, args)
    (root / "report.md").write_text(rerank_report(root, benchmark))
