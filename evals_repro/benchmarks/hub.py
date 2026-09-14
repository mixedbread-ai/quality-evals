import json
from collections.abc import Iterator

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def rows(source: tuple[str, str], filename: str) -> Iterator[dict]:
    repo, revision = source
    path = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
    if filename.endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    else:
        with open(path) as stream:
            yield from (json.loads(line) for line in stream)
