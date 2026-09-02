import argparse
from pathlib import Path

from dotenv import load_dotenv

from evals_repro.benchmarks import Benchmark, registry
from evals_repro.bm25 import BM25Retriever
from evals_repro.data import Setting, Subset
from evals_repro.embedders.registry import EMBEDDERS
from evals_repro.evaluate import evaluate_subset, result_path
from evals_repro.report import report
from evals_repro.rerank import RERANKERS, RerankedRetriever
from evals_repro.retrievers import DenseRetriever, Retriever
from evals_repro.stores import StoreRetriever, StoreUploader

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = registry()
STORE_METHODS = ("mixedbread-markdown", "mixedbread-markdown+rerank", "mixedbread-images", "mixedbread-images+rerank")


def store_retriever(method: str, prefix: str, rerank_top_k: int) -> StoreRetriever:
    content, _, reranked = method.removeprefix("mixedbread-").partition("+")
    return StoreRetriever(prefix, content, rerank_top_k if reranked else None)


def store_prefix(args: argparse.Namespace, benchmark: Benchmark) -> str:
    return args.store_prefix or f"{benchmark.name}-eval"


def build_retriever(args: argparse.Namespace, cache_dir: Path, prefix: str) -> Retriever:
    if args.method == "bm25":
        retriever: Retriever = BM25Retriever()
    elif args.method in STORE_METHODS:
        retriever = store_retriever(args.method, prefix, args.rerank_top_k)
    else:
        embedder = EMBEDDERS[args.method]()
        if args.content == "images" and not hasattr(embedder, "embed_images"):
            raise SystemExit(f"{args.method} cannot embed images")
        retriever = DenseRetriever(embedder, cache_dir, args.content)
    if args.content == "images" and args.method not in EMBEDDERS:
        raise SystemExit("--content applies to dense embedders only")
    if args.rerank and args.method in STORE_METHODS:
        raise SystemExit("Mixedbread stores rerank with their own model; use mixedbread-*+rerank")
    if args.rerank:
        retriever = RerankedRetriever(retriever, RERANKERS[args.rerank](), args.rerank_depth)
    return retriever


def languages_for(subset: Subset, setting: Setting, benchmark: Benchmark) -> list[str]:
    if setting == "monolingual" or not benchmark.crosslingual:
        return [subset.native_language]
    return list(benchmark.languages)


def run(args: argparse.Namespace) -> None:
    benchmark = BENCHMARKS[args.benchmark]
    results_dir = args.results_dir / benchmark.name
    retriever = build_retriever(args, args.cache_dir / benchmark.name, store_prefix(args, benchmark))
    for name in args.subsets or benchmark.subsets:
        if args.resume and result_path(results_dir, retriever.name, name).exists():
            continue
        subset = benchmark.load(name)
        result = evaluate_subset(retriever, subset, languages_for(subset, args.setting, benchmark), args.top_k)
        path = result.save(results_dir)
        scores = " ".join(f"{lr.language}={100 * lr.metrics['ndcg_cut_10']:.2f}" for lr in result.languages)
        print(f"{result.method} {name}: {scores}  -> {path}")


def upload(args: argparse.Namespace) -> None:
    benchmark = BENCHMARKS[args.benchmark]
    uploader = StoreUploader(store_prefix(args, benchmark), args.parsing, args.workers)
    for name in args.subsets or benchmark.subsets:
        store = uploader.upload(benchmark.load(name), args.content)
        print(f"{store}: {uploader.wait_processed(store)}")


def show(args: argparse.Namespace) -> None:
    benchmark = BENCHMARKS[args.benchmark]
    print(report(args.results_dir / benchmark.name, benchmark, args.measure, args.first_stage_only))


def main() -> None:
    load_dotenv(ROOT / ".env")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--benchmark", choices=list(BENCHMARKS), default="vidore-v3")
    shared.add_argument("--results-dir", type=Path, default=ROOT / "results")

    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--subsets", nargs="+")
    selection.add_argument("--store-prefix")

    parser = argparse.ArgumentParser(prog="evals-repro")
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", parents=[shared, selection])
    runner.add_argument("method", choices=["bm25", *STORE_METHODS, *EMBEDDERS])
    runner.add_argument("--setting", choices=["monolingual", "crosslingual"], default="crosslingual")
    runner.add_argument("--content", choices=["markdown", "images"], default="markdown")
    runner.add_argument("--top-k", type=int, default=100)
    runner.add_argument("--rerank", choices=list(RERANKERS))
    runner.add_argument("--rerank-depth", type=int, default=50)
    runner.add_argument("--rerank-top-k", type=int, default=10)
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "embeddings")
    runner.set_defaults(func=run)

    uploader = commands.add_parser("mixedbread-upload", parents=[shared, selection])
    uploader.add_argument("content", choices=["markdown", "images"])
    uploader.add_argument("--parsing", choices=["fast", "high_quality"], default="fast")
    uploader.add_argument("--workers", type=int, default=8)
    uploader.set_defaults(func=upload)

    reporter = commands.add_parser("report", parents=[shared])
    reporter.add_argument("--measure", default="ndcg_cut_10")
    reporter.add_argument("--first-stage-only", action="store_true")
    reporter.set_defaults(func=show)

    args = parser.parse_args()
    args.func(args)
