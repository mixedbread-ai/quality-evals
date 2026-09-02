<p align="center">
  <a href="https://www.mixedbread.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://www.mixedbread.com/images/brand/logos/wordmark_dark.svg">
      <img alt="Mixedbread" src="https://www.mixedbread.com/images/brand/logos/wordmark_light.svg" width="280">
    </picture>
  </a>
</p>

# evals-repro

*This repository is by-agent-for-agents, to automatically reproduce the evaluation results we report on [our evaluation page](https://www.mixedbread.com/evals). It is the exact code used to obtain the numbers we reported, and is designed for you to point your agent at it.*

Sample code for reproducing the retrieval numbers on our evaluation page. One data loader, one set of qrels and one
scorer (`pytrec_eval`) sit behind every method, so the systems being compared differ only in how they retrieve.

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
uv sync
cp .env.example .env   # VOYAGE_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, MXBAI_API_KEY
```

Benchmark data is pulled from the Hugging Face hub on first use.

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

## Layout

| module | role |
|---|---|
| `data.py` | `Subset`, `Query`, consistency checks |
| `benchmarks/` | `Benchmark` registry; `vidore_v3.py` and `miracl_vision.py` load subsets from parquet |
| `index.py` | `ExactIndex`, exact cosine top-k with deterministic tie-breaking |
| `metrics.py` | `pytrec_eval` wrapper, macro averages |
| `embedders/` | `Embedder` protocol, budgeted batching, Voyage, Voyage multimodal, Cohere and Gemini clients |
| `retrievers.py` | `Retriever` protocol, `DenseRetriever` over markdown or page images |
| `bm25.py` | `BM25Retriever` |
| `stores.py` | Mixedbread `StoreUploader` and `StoreRetriever` |
| `rerank.py` | `RerankedRetriever` with the Cohere reranker |
| `cache.py`, `throttle.py`, `clients.py` | embedding cache, per-minute budgets, API client construction |
| `evaluate.py` | per-subset evaluation and result files |
| `report.py` | markdown tables |
| `cli.py` | `evals-repro` entry point |
