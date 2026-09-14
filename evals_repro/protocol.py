import json
import random
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from evals_repro.benchmarks import Benchmark
from evals_repro.cache import digest
from evals_repro.data import Subset
from evals_repro.records import fingerprint


def timing_counts(benchmark: str, names: list[str], samples: int, seed: int) -> dict[str, int]:
    if benchmark == "text":
        return dict.fromkeys(names, samples)
    order = sorted(names)
    random.Random(f"{seed}:{benchmark}:allocation").shuffle(order)
    count, remainder = divmod(samples, len(order))
    return {name: count + (i < remainder) for i, name in enumerate(order)}


def manifest_for(args, benchmark: Benchmark, root: Path) -> dict:
    config = {key: getattr(args, key) for key in ("depth", "seed", "samples", "repeats", "first_stage", "store_prefix")}
    config["implementation"] = {
        p: version(p) for p in ("bm25s", "pystemmer", "numpy", "pytrec-eval-terrier", "pyndeval")
    }
    config.update(benchmark=benchmark.name, subsets=args.subsets or list(benchmark.subsets), setting="monolingual")
    config["timing_queries"] = timing_counts(benchmark.name, config["subsets"], args.samples, args.seed)
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["protocol"] != 1 or manifest["config"] != config:
            raise ValueError("Study configuration changed; use a new --study-dir")
    else:
        manifest = {
            "config": config,
            "created_at": datetime.now(UTC).isoformat(),
            "models": {},
            "datasets": {},
            "selections": {},
            "sessions": [],
            "protocol": 1,
        }
    return manifest


def register_subset(manifest: dict, subset: Subset) -> None:
    metadata = {
        "source": subset.source,
        "language": subset.native_language,
        "documents": len(subset.pages),
        "queries": len(subset.queries),
        "corpus_sha256": digest(subset.pages.items()),
        "queries_sha256": fingerprint(sorted((q.id, q.text, q.language) for q in subset.queries)),
        "qrels_sha256": fingerprint(subset.qrels),
        "annotations_sha256": fingerprint(subset.annotations),
        "retrieval_text_sha256": fingerprint(subset.retrieval_text),
    }
    previous = manifest["datasets"].setdefault(subset.name, metadata)
    if previous != metadata:
        raise ValueError(f"{subset.name}: dataset changed; use a new --study-dir")
