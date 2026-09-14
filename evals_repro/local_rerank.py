import os
from dataclasses import dataclass, field
from queue import Queue
from threading import local

import httpx
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


@dataclass(frozen=True)
class LocalModel:
    model: str
    revision: str
    prefix: str
    separator: str
    suffix: str
    classifier: tuple[str, ...]
    method: str
    urls_env: str
    port: int
    context: int = 32768

    def prompt(self, query: str, document: str = "") -> str:
        return self.prefix + query + self.separator + document + self.suffix

    @property
    def template(self) -> str:
        import json

        query = 'messages | selectattr("role", "eq", "query") | map(attribute="content") | first'
        document = 'messages | selectattr("role", "eq", "document") | map(attribute="content") | first'
        return "".join(
            "{{ " + value + " }}"
            for value in (
                json.dumps(self.prefix),
                query,
                json.dumps(self.separator),
                document,
                json.dumps(self.suffix),
            )
        )


LOCAL_MODELS = {
    "qwen3-8b": LocalModel(
        "Qwen/Qwen3-Reranker-8B",
        "77d193c791ed757ca307ee72715aa132723da912",
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: ",
        "\n<Document>: ",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ("no", "yes"),
        "from_2_way_softmax",
        "QWEN_RERANK_URLS",
        8300,
    ),
    "zerank-2": LocalModel(
        "zeroentropy/zerank-2",
        "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c",
        "<|im_start|>system\n",
        "<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n",
        ("Yes",),
        "no_post_processing",
        "ZERANK_RERANK_URLS",
        8400,
    ),
}


@dataclass
class LocalReranker:
    key: str
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=600))
    endpoints: Queue = field(default_factory=Queue)
    timing_context: local = field(default_factory=local)

    def __post_init__(self) -> None:
        self.spec = LOCAL_MODELS[self.key]
        self.tokenizer = Tokenizer.from_file(hf_hub_download(self.model, "tokenizer.json", revision=self.spec.revision))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        urls = os.environ.get(self.spec.urls_env, f"http://127.0.0.1:{self.spec.port}").split(",")
        self.urls = tuple(url.strip().rstrip("/") for url in urls)
        for url in self.urls:
            self.endpoints.put(url)

    @property
    def provider(self) -> str:
        return self.key

    @property
    def model(self) -> str:
        return self.spec.model

    @property
    def name(self) -> str:
        return "qwen3-reranker-8b" if self.key == "qwen3-8b" else self.key

    @property
    def metadata(self) -> dict:
        return {
            "revision": self.spec.revision,
            "dtype": "bfloat16",
            "context": self.spec.context,
            "prompt": self.spec.prompt("{query}", "{document}"),
            "scoring": self.spec.method,
        }

    def validate_latency(self) -> dict:
        checked = {}
        for url in self.urls:
            response = self.client.get(url + "/metrics")
            response.raise_for_status()
            configs = [line for line in response.text.splitlines() if line.startswith("vllm:cache_config_info{")]
            if not configs or any('enable_prefix_caching="False"' not in line for line in configs):
                raise ValueError(f"{url}: cannot verify prefix caching is disabled; restart with serve-reranker")
            checked[url] = {"prefix_caching": False}
        return checked

    def prepare(self, query: str, documents: list[str]) -> list[str]:
        budget = (
            self.spec.context - len(self.tokenizer.encode(self.spec.prompt(query), add_special_tokens=False).ids) - 8
        )
        if budget < 1:
            raise ValueError(f"Query exceeds {self.name}'s context")
        prepared, truncated = [], 0
        for document in documents:
            ids = self.tokenizer.encode(document, add_special_tokens=False).ids
            if len(ids) > budget:
                document = self.tokenizer.decode(ids[:budget], skip_special_tokens=False)
                truncated += 1
            prepared.append(document)
        self.timing_context.truncated_documents = truncated
        return prepared

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        from evals_repro.rerank import scores_by_position

        documents = self.prepare(query, documents)
        endpoint = self.endpoints.get()
        try:
            response = self.client.post(
                endpoint + "/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                    "use_activation": self.spec.method == "from_2_way_softmax",
                },
            )
            response.raise_for_status()
            return scores_by_position(response.json()["results"], len(documents))
        finally:
            self.endpoints.put(endpoint)
