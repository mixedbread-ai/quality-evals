import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import numpy as np
from mixedbread import APIConnectionError
from voyageai.error import APIConnectionError as VoyageConnectionError
from voyageai.error import Timeout as VoyageTimeout

from evals_repro.rerank import Reranker
from evals_repro.throttle import Budget


def retry_after(error: Exception, attempt: int) -> float | None:
    response = getattr(error, "response", None)
    status = (
        getattr(error, "status_code", None)
        or getattr(error, "http_status", None)
        or getattr(response, "status_code", None)
    )
    headers = getattr(response, "headers", None) or getattr(error, "headers", {}) or {}
    transient = isinstance(error, (httpx.TransportError, APIConnectionError, VoyageConnectionError, VoyageTimeout))
    if status not in (408, 429, 500, 502, 503, 504) and not transient:
        return None
    value = headers.get("retry-after")
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return max(60 if status == 429 else 0, min(60, 2**attempt))


@dataclass
class MeasuredReranker:
    reranker: Reranker
    interval: float = 0
    max_retries: int = 12
    next_request: float = field(default=0, init=False)
    token_budget: Budget | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def wait(self) -> None:
        while True:
            with self.lock:
                delay = self.next_request - time.monotonic()
                if delay <= 0:
                    self.next_request = time.monotonic() + self.interval
                    return
            time.sleep(delay)

    def run(self, query: str, documents: list[str]) -> tuple[list[float], dict]:

        attempts = []
        started = time.perf_counter_ns()
        for attempt in range(self.max_retries + 1):
            if self.token_budget is not None:
                self.token_budget.reserve((len(query) * len(documents) + sum(map(len, documents)) + 2) // 3)
            self.wait()
            stamp = datetime.now(UTC).isoformat()
            before = time.perf_counter_ns()
            try:
                scores = self.reranker.rerank(query, documents)
            except Exception as error:
                elapsed = time.perf_counter_ns() - before
                delay = retry_after(error, attempt)
                if delay is None or attempt == self.max_retries:
                    raise
                status = (
                    getattr(error, "status_code", None)
                    or getattr(error, "http_status", None)
                    or getattr(getattr(error, "response", None), "status_code", None)
                )
                attempts.append(
                    {
                        "started_at": stamp,
                        "elapsed_ns": elapsed,
                        "status": status,
                        "error": type(error).__name__,
                        "retry_wait_s": delay,
                    }
                )
                print(f"{self.reranker.name}: {type(error).__name__}; retry in {delay:.1f}s", flush=True)
                if status == 429:
                    with self.lock:
                        self.next_request = max(self.next_request, time.monotonic() + delay)
                time.sleep(delay)
                continue
            elapsed = time.perf_counter_ns() - before
            if len(scores) != len(documents) or not np.isfinite(scores).all():
                raise ValueError(f"{self.reranker.name}: incomplete or nonfinite rerank scores")
            attempts.append({"started_at": stamp, "elapsed_ns": elapsed, "status": 200})
            return scores, {"latency_ns": elapsed, "wall_ns": time.perf_counter_ns() - started, "attempts": attempts}
        raise AssertionError("unreachable")


def statistics(records: list[dict], seed: int = 42, bootstraps: int = 2000, clock: str = "latency_ns") -> dict:

    if not records:
        raise ValueError("No latency measurements")
    groups: dict[str, dict[str, list[float]]] = {}
    for row in records:
        groups.setdefault(row["subset"], {}).setdefault(row["query_id"], []).append(row[clock] / 1e6)
    strata = [[groups[s][qid] for qid in sorted(groups[s])] for s in sorted(groups)]
    values = np.array([v for queries in strata for repeats in queries for v in repeats])

    def measures(x):
        return [np.mean(x), *np.percentile(x, [50, 90, 95])]

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstraps):
        sampled = [v for queries in strata for i in rng.integers(len(queries), size=len(queries)) for v in queries[i]]
        samples.append(measures(sampled))
    intervals = np.percentile(samples, [2.5, 97.5], axis=0).T
    return {
        "requests": len(records),
        "queries": sum(len(queries) for queries in strata),
        "retried_requests": sum(len(row["attempts"]) > 1 for row in records),
        "failed_attempts": sum(len(row["attempts"]) - 1 for row in records),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        **{
            name: {"ms": float(value), "ci95_ms": interval.tolist()}
            for name, value, interval in zip(("mean", "p50", "p90", "p95"), measures(values), intervals, strict=True)
        },
    }


def repeat_effect(records: list[dict], seed: int = 42, bootstraps: int = 2000) -> dict:
    groups = {}
    for row in records:
        if row["repeat"] < 0:
            continue
        repeats = groups.setdefault((row["subset"], row["query_id"]), {})
        if row["repeat"] in repeats:
            raise ValueError("Duplicate latency repetition")
        repeats[row["repeat"]] = row
    strata = {}
    for (subset, _), repeats in sorted(groups.items()):
        first = repeats.get(0)
        later = [row["latency_ns"] / 1e6 for r, row in repeats.items() if r > 0 and len(row["attempts"]) == 1]
        if first and first["latency_ns"] > 0 and len(first["attempts"]) == 1 and later:
            strata.setdefault(subset, []).append((first["latency_ns"] / 1e6, float(np.median(later))))
    pairs = [pair for rows in strata.values() for pair in rows]
    result = {"paired_queries": len(pairs), "excluded_queries": len(groups) - len(pairs)}
    if not pairs:
        return {**result, "status": "insufficient_queries"}
    first, later = np.array(pairs).T
    result.update(first_p50_ms=float(np.median(first)), repeat_p50_ms=float(np.median(later)))
    if len(pairs) < 5:
        return {**result, "status": "insufficient_queries"}
    rng = np.random.default_rng(seed)
    sampled = []
    for rows in strata.values():
        values = np.array([later / first for first, later in rows])
        sampled.append(values[rng.integers(len(values), size=(bootstraps, len(values)))])
    low, high = np.percentile(np.median(np.concatenate(sampled, axis=1), axis=1), [2.5, 97.5])
    return {
        **result,
        "ratio": {"estimate": float(np.median(later / first)), "ci95": [float(low), float(high)]},
        "status": "speedup" if high < 0.9 else "no_material_speedup" if low >= 0.9 else "inconclusive",
    }
