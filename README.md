<p align="center">
  <a href="https://www.mixedbread.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://www.mixedbread.com/images/brand/logos/wordmark_dark.svg">
      <img alt="Mixedbread" src="https://www.mixedbread.com/images/brand/logos/wordmark_light.svg" width="280">
    </picture>
  </a>
</p>

# evals-repro

Reproduce retrieval scores, reranking quality, and reranking latency reported on [our evaluation page](https://www.mixedbread.com/evals).
Methods share dataset loaders, relevance judgments, and scoring. Reranking comparisons use identical frozen candidate documents.

| Reproduction | Commands | Documentation |
|---|---|---|
| Retrieval and ViDoRe v3 store scores | `mixedbread-upload`, `run`, `report` | [Retrieval](#running), [ViDoRe stores](#vidore-v3-store-scores) |
| Reranking quality | `rerank-study prepare`, `quality`, `report` | [Reranking quality](#reranking-quality) |
| Reranking timings | `rerank-study latency`, `report` | [Reranking timing](#reranking-timing) |

Benchmarks are a parameter. Each one declares its subsets, its query languages and how to load a subset (`benchmarks/`):

- `vidore-v3` ([ViDoRe v3](https://huggingface.co/collections/vidore/vidore-benchmark-v3)): eight specialised corpora,
  pages available as OCR markdown or as images, queries in six languages, graded relevance
- `miracl-vision` ([MIRACL-VISION](https://huggingface.co/datasets/nvidia/miracl-vision)): Wikipedia article images in
  eighteen languages, one subset per language, queries in the language of the corpus. Every method sees the article
  title next to its image (`title: …` as a text part of the embedding input, file metadata for Mixedbread stores)

Methods share one interface (`Retriever.run(subset, queries, top_k) -> Run`):

- dense embeddings over an exact (brute-force cosine) index, on page markdown or page images:
  `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `cohere-embed-v4.0`, `voyage-multimodal-3.5`, `gemini-embedding-2`
  (`gemini-embedding-2-noprompt` sends bare text instead of the prompt prefixes Google documents)
- `bm25` (bm25s, Snowball stemmer of the corpus language)
- Mixedbread Stores (`mixedbread-markdown`, `mixedbread-images`, each optionally `+rerank`): page markdown or page images
  uploaded to a store, searched with and without the listwise reranker
- optionally, any of the above followed by a third-party reranker (`rerank.py`, kept separate from the first-stage code)

## Setup

```sh
uv sync --locked --python 3.12
cp .env.example .env   # VOYAGE_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MXBAI_API_KEY
```

Benchmark data is downloaded on first use. Public Hugging Face snapshots are pinned by revision; MS MARCO downloads
use `ir_datasets` checksums. Rebuilding the full MS MARCO passage index requires substantial RAM and disk space.

## Running

```sh
uv run evals-repro run voyage-4-large                                   # every subset, crosslingual
uv run evals-repro run bm25 --setting monolingual                       # queries in the corpus language only
uv run evals-repro run voyage-multimodal-3.5 --content images           # page images as documents
uv run evals-repro run voyage-4-large --rerank cohere                   # rerank the top 50 with rerank-v4.0-pro
uv run evals-repro mixedbread-upload markdown --subsets computer_science   # create stores, upload pages
uv run evals-repro mixedbread-upload images --parsing high_quality
uv run evals-repro run mixedbread-markdown                              # store search
uv run evals-repro run mixedbread-images+rerank                         # store search + listwise reranker
uv run evals-repro run voyage-multimodal-3.5 --benchmark miracl-vision --content images
uv run evals-repro report
```

Shared options: `--benchmark` (default `vidore-v3`), `--results-dir`. `run` options: `--subsets`,
`--setting monolingual|crosslingual`, `--content markdown|images` (dense embedders only), `--top-k` (default 100),
`--rerank cohere` with `--rerank-depth` (default 50), `--rerank-top-k` for the Mixedbread reranker (default 10), `--resume`
to skip subsets that already have a result file, `--cache-dir`, `--store-prefix` (default `<benchmark>-eval`). `mixedbread-upload` options: `--subsets`,
`--parsing fast|high_quality`, `--workers`, `--store-prefix`. `report` options: `--measure`, `--first-stage-only`.

Results land in `results/<benchmark>/<method>/<subset>.json` with per-language and per-query metrics; reranked methods
are named `<first stage>+<reranker>@<depth>`. `report` prints a *monolingual* table (queries in each corpus's own
language) and a *crosslingual* table (average over all query languages of the benchmark). The `avg` column is filled
only when every subset is present.

Embeddings are cached under `cache/embeddings/<benchmark>/<method>/`, keyed by a digest of the full text or image list,
so an unchanged subset is never re-embedded. Store uploads are resumable: pages already present in a store are skipped.
Reranked runs score only the reranked candidates, so their `recall_*` values beyond the rerank depth are not meaningful.

### ViDoRe v3 store scores

Upload the pinned ViDoRe pages, run store retrieval, and report monolingual and crosslingual scores:

```sh
uv run evals-repro mixedbread-upload markdown --benchmark vidore-v3 --store-prefix vidore-eval
uv run evals-repro run mixedbread-markdown --benchmark vidore-v3 --store-prefix vidore-eval --setting crosslingual
uv run evals-repro run mixedbread-markdown+rerank --benchmark vidore-v3 --store-prefix vidore-eval --setting crosslingual
uv run evals-repro report --benchmark vidore-v3
```

Use `--setting monolingual` to evaluate only each corpus's native-language queries. For image stores, upload `images`
and run `mixedbread-images` or `mixedbread-images+rerank` with the same prefix. Store retrieval has its own `run` and
`report` workflow, with results under `results/vidore-v3/`. The reranking studies below use separate study directories.

## Layout

| module | role |
|---|---|
| `data.py` | `Subset`, `Query`, consistency checks |
| `benchmarks/` | Benchmark registry and pinned dataset loaders |
| `index.py` | `ExactIndex`, exact cosine top-k with deterministic tie-breaking |
| `metrics.py` | `pytrec_eval` wrapper, macro averages |
| `embedders/` | `Embedder` protocol, budgeted batching, Voyage, Voyage multimodal, Cohere and Gemini clients |
| `retrievers.py` | `Retriever` protocol, `DenseRetriever` over markdown or page images |
| `bm25.py` | `BM25Retriever` |
| `stores.py` | Mixedbread `StoreUploader` and `StoreRetriever` |
| `rerank.py` | Reranker clients and `RerankedRetriever` |
| `study.py`, `protocol.py`, `candidates.py`, `records.py` | Reproducible phases, query allocation, frozen candidates, resumable records |
| `latency.py` | Serial request timing, explicit retries, query-cluster intervals |
| `local_rerank.py`, `serve.py` | Pinned local rerankers and GPU serving |
| `overview.py` | Benchmark-group averages, dataset breakdowns, CSV export, and pooled timings |
| `cache.py`, `throttle.py`, `clients.py` | embedding cache, per-minute budgets, API client construction |
| `evaluate.py` | per-subset evaluation and result files |
| `report.py` | markdown tables |
| `cli.py` | `evals-repro` entry point |

## Reranker quality and timing

`rerank-study` retrieves the top 100 documents per query and evaluates every reranker against the same frozen candidates.
ViDoRe uses Markdown stores by default; the other benchmarks use BM25 indexes built with `bm25s`. Preparation, quality evaluation, and serial timing can run independently and resume.
The default benchmark selection runs every collection below in its native query language.

| Benchmark | Subsets | Evaluation |
|---|---|---|
| `text` | `fiqa`, `trec-covid`, `dl19`, `dl20`, `hotpotqa`, `nq`, `scifact` | nDCG@10, MAP, recall |
| `financebench` | `financebench` | 150 queries against the released 145-document evidence corpus |
| `legalragbench` | `legalragbench` | 100 expert questions against 4,876 released Markdown passages |
| `cure` | Ten medical specialties | Full CUREv1 English queries; a separate corpus and index per specialty |
| `freshstack` | `langchain`, `yolo`, `laravel`, `angular`, `godot` | Full October 2024 release; α-nDCG@10, α=0.5, and nugget coverage@20 |
| `followir` | `robust04`, `core17`, `news21` | Full instruction pairs; MAP, nDCG@10, and p-MRR |
| `vidore-v3` | All eight subsets | Full page Markdown; native-language queries; nDCG@10 |

Dataset revisions are pinned in the loaders. TREC DL19 and DL20 use the judged passage tasks from `ir_datasets` and
share the full MS MARCO passage index. The lockfile pins `ir_datasets`, whose downloads validate source checksums.
BEIR document text combines title and body. FreshStack queries contain the question title and body; nugget annotations
are used only for scoring. LegalRAGBench queries contain the question. FinanceBench evaluates the released evidence
corpus. ViDoRe reranking uses the Markdown column of the pinned corpus parquet files.

FollowIR retrieves with the original query text and sends the query plus its instruction to the reranker. Both instruction
variants receive identical candidates. This is a BM25 top-100 adaptation: MAP is computed over the full returned ranking,
and p-MRR assigns missing changed documents rank 101. Its results are specific to this candidate pool. FreshStack uses
`pyndeval` for α-nDCG; relevance annotations never select candidates. All query means include zero-scoring queries.

### Reranking quality

```sh
uv sync --locked --python 3.12
uv run --locked evals-repro mixedbread-upload markdown --benchmark vidore-v3 --store-prefix vidore-eval
uv run --locked evals-repro rerank-study prepare
uv run --locked evals-repro rerank-study quality
uv run --locked evals-repro rerank-study report
```

The upload command creates resumable Markdown stores from pinned public ViDoRe data using your Mixedbread account.
Preparation rebuilds missing BM25 indexes and freezes candidates for every evaluation query. Quality evaluation scores
every query with the four API models by default. Reports include the first-stage baseline and each reranker.
These commands use `results/rerank-study/`; repeat them with the same options to resume completed work.

Set `MXBAI_API_KEY`, `VOYAGE_API_KEY`, and `COHERE_API_KEY` in the environment or the repository's `.env`.
Select `--models mixedbread`, for example, to run one service with its key. ViDoRe store creation and candidate retrieval
require a Mixedbread key. BM25 preparation and report generation require no API credentials. Use `--benchmark` for one
collection and `--subsets` for selected subsets within it. Available models are:

| CLI name | Model |
|---|---|
| `mixedbread` | `mixedbread-ai/mxbai-rerank-v3.1-listwise`, result label `mixedbread_prod_260910` |
| `voyage` | `rerank-3` |
| `cohere-pro` | `rerank-v4.0-pro` |
| `cohere-fast` | `rerank-v4.0-fast` |
| `qwen3-8b` | `Qwen/Qwen3-Reranker-8B`, pinned revision |
| `zerank-2` | `zeroentropy/zerank-2`, pinned revision |

Select models with `--models mixedbread voyage cohere-pro`, for example. Additional models can be evaluated in the
same study directory; completed requests are reused. Provider model identifiers are saved in the manifest, but a hosted
provider can update the deployment behind an identifier. The saved scores and request timestamps identify each run.

### Local models

The `gpu` dependency group installs pinned vLLM on Linux with Python 3.12 or 3.13 and a compatible NVIDIA driver.
Start the servers in separate terminals, assigning each an available GPU:

```sh
uv run --locked --group gpu evals-repro serve-reranker qwen3-8b --devices 0
uv run --locked --group gpu evals-repro serve-reranker zerank-2 --devices 1
```

Then run quality evaluation against them:

```sh
uv run --locked evals-repro rerank-study quality --models qwen3-8b zerank-2
```

The servers bind to localhost ports 8300 and 8400 respectively. Multiple `--devices` start independent replicas on
consecutive ports. Set `QWEN_RERANK_URLS` or `ZERANK_RERANK_URLS` to comma-separated base URLs to use those replicas
or remote hosts for quality evaluation.

`local_rerank.py` pins each checkpoint, tokenizer, chat format, and scoring rule. Both run in BF16 with 32,768-token
contexts. Qwen uses the original no/yes softmax probability; ZeRank uses the raw `Yes` logit. Local clients preserve the
complete query and prompt, truncate only the document suffix when needed, and record the number of truncated documents.
Serving commands are generated from the same model specifications used by the clients. Prefix caching is disabled.
Local models require a compatible GPU with sufficient memory; API credentials alone do not provide local inference.

### Reranking timing

Choose a new study directory for each measurement date or hosted model deployment. Reusing a directory resumes its
saved measurements. Timing prepares sampled candidates using the same first-stage defaults as quality evaluation.
Create the ViDoRe stores with the upload command above before running the full suite:

```sh
uv run --locked evals-repro rerank-study latency --samples 20 --repeats 5 --seed 42 \
  --study-dir results/rerank-study-timing --host-location "provider, region, machine"
uv run --locked evals-repro rerank-study report --study-dir results/rerank-study-timing
```

Run timing with other reranking jobs paused on the same credentials and host, so quality concurrency does not affect
the latency measurements.

Timing defaults to the four API models and 20 queries per dataset: each of the seven BEIR/TREC datasets, FinanceBench,
LegalRAGBench, CURE, FreshStack, FollowIR, and ViDoRe. For a benchmark with several subsets, the 20 queries are allocated
as evenly as possible across its subsets, with seed 42 assigning any remainder. CURE therefore samples two queries per
specialty; FreshStack samples four per topic. Queries are sampled without replacement from sorted IDs using the seed and
subset name. Selections are saved before requests begin. Every query runs five times, with deterministic shuffling of
query and model order. The complete default comparison contains 5,200 measured calls across 13 datasets and four models.
Timing is serial: only one reranker call is active. One warmup per model and subset per invocation is saved separately
and excluded from the measurements. To run ten repetitions, use a separate study directory:

```sh
uv run --locked evals-repro rerank-study latency --repeats 10 --study-dir results/rerank-study-10x --host-location "provider, region, machine"
```

`--samples` and `--seed` control selection; `--subsets` distributes a family's query allowance across just those subsets.
`--quality-workers` controls concurrent quality requests per provider (default 2). Each provider has its own worker pool,
so a Voyage budget wait cannot occupy a Mixedbread or Cohere worker. Local models have separate pools. API calls use
persistent clients with SDK retries disabled; explicit retries honor `Retry-After` and record every attempt. Voyage requests share a
per-minute estimated token budget (`--voyage-tpm`, default 2,000,000) and use `--voyage-interval` seconds between calls
(default 3). Token estimates use characters; provider rate-limit responses remain authoritative. A Voyage batch exceeding
its aggregate token limit is split into smaller batches while retaining every document and its position. Cohere uses
`max_tokens_per_doc=32768`. API requests otherwise receive the full candidate text.

Successful-call latency covers the complete reranker client call, including network, response parsing, and any local
preprocessing or batch splitting. It is measured with `perf_counter_ns`. Wall time additionally includes failed attempts,
backoff, and scheduling waits. Both have mean, p50, p90, p95, and 95% percentile bootstrap intervals from 2,000 resamples
of whole query clusters, stratified by subset. Repetitions of one query stay together. These are client-observed timings;
quality-run concurrency is not used to report latency benchmarks.

Reports also compare each query's first measured call with the median of its later calls. A paired bootstrap, stratified
by subset, estimates the median repeat/first ratio and its 95% interval. An interval entirely below 0.90 flags a repeat
speedup exceeding 10%; an interval crossing 0.90 is inconclusive. At least five query pairs are required. Retried calls
and warmups are excluded from this diagnostic. Inspect flagged runs before using their pooled timing estimates.
First measured calls can already be warm, and a speedup may have causes besides caching; this check cannot certify that
a hosted API is uncached. The report includes first-pass and repeat medians so the effect remains visible.

For optional local timing, select `--models qwen3-8b zerank-2`. Before any measurement, the client verifies disabled prefix
caching through every vLLM replica's `/metrics` endpoint and records the checks in the session manifest. Enabled or
unverifiable caching stops the run. Use `serve-reranker` to configure these servers and a new study directory when the
server checks change, including when resuming measurements made without this validation.

### Study artifacts

Generated candidates, scores, request timings, and reports stay in the local, Git-ignored `results/` directory.

Artifacts live under `results/rerank-study/<benchmark>/monolingual/`:

| Artifact | Contents |
|---|---|
| `manifest.json` | Protocol, dataset hashes, revisions, selected query IDs, model configuration, packages, host, sessions |
| `candidates/<subset>.jsonl` | Ordered candidate IDs, first-stage scores, request and candidate digests |
| `quality.jsonl` | Per-query reranker scores and request attempts |
| `latency.jsonl`, `warmup.jsonl` | Measured repetitions and separately retained warmups |
| `quality/<method>/<subset>.json` | Aggregate and per-query quality metrics, including the first-stage baseline |
| `latency-summary.json`, `report.md` | Timing distributions, confidence intervals, repeat-speedup checks, coverage, quality tables |

BM25 uses Lucene scoring, `k1=1.5`, `b=0.75`, a Snowball stemmer for the corpus language, and no stopword removal.
Indexes are saved under `cache/bm25/` with corpus and implementation fingerprints. A changed fingerprint rebuilds the
index. Study locks prevent simultaneous writers; each completed request is flushed to disk, and an interrupted final
JSONL write is recovered on resume. Changed datasets, candidates, or protocol settings require a new study directory.
Quality table averages appear only when all selected subsets have complete results.

With the default `--benchmark all`, `rerank-study report` also writes `report.md`, `quality.csv`, and, when timing records
exist, `latency-summary.json` at the study-directory root. The CSV includes every dataset/subset, its query count, metric,
first stage, baseline, and reranker scores. Benchmark means weight their subsets equally. The overall quality score
weights the seven benchmark groups equally: the seven BEIR/TREC datasets together form the `text` group. FreshStack
contributes alpha-nDCG@10; the other groups contribute nDCG@10. Scores are multiplied by 100. Incomplete means are blank.

Overall timing pools all measured requests across the 13 datasets, with 20 queries and the same number of repetitions
per dataset. Its p50 is the median of those requests. Host and protocol settings must match across the suite, and an
incomplete run has no overall p50. Individual benchmark reports retain the distributions and repeat-effect checks.

The code reproduces the inputs, scoring, and measurement protocol. Hosted model deployments, store search results,
network routes, and service load can change; historical numerical results require the corresponding deployment and
frozen inputs. Use separate study directories for deployment comparisons and retain their manifests locally.

ViDoRe candidate retrieval disables store reranking; each selected rerank endpoint scores the frozen Markdown pages.
To choose the first stage explicitly:

```sh
uv run --locked evals-repro rerank-study quality --benchmark vidore-v3 --first-stage mixedbread-markdown --study-dir results/rerank-study-stores
uv run --locked evals-repro rerank-study quality --benchmark vidore-v3 --first-stage bm25 --study-dir results/rerank-study-bm25
```
