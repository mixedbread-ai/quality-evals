import json
from dataclasses import replace

import httpx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from evals_repro.local_rerank import LOCAL_MODELS, LocalReranker


@pytest.fixture
def tokenizer_file(tmp_path, monkeypatch):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "question": 1, "document": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    monkeypatch.setattr("evals_repro.local_rerank.hf_hub_download", lambda *a, **kw: str(path))


@pytest.mark.parametrize("key", list(LOCAL_MODELS))
def test_local_endpoint_preserves_positions_and_scoring_rule(key, tokenizer_file):
    def respond(request):
        payload = json.loads(request.content)
        assert payload["query"] == "question"
        assert payload["documents"] == ["document", "unrelated"]
        assert payload["use_activation"] == (key == "qwen3-8b")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": -4.5},
                    {"index": 0, "relevance_score": 5.4},
                ]
            },
        )

    runner = LocalReranker(key, client=httpx.Client(transport=httpx.MockTransport(respond)))
    assert runner.rerank("question", ["document", "unrelated"]) == [5.4, -4.5]


def test_context_limit_preserves_query_and_records_document_truncation(tokenizer_file):
    runner = LocalReranker("zerank-2")
    runner.spec = replace(runner.spec, context=64)
    query = "question " * 10
    documents = runner.prepare(query, ["document " * 100, "document"])
    assert documents[1] == "document" and runner.timing_context.truncated_documents == 1
    assert query in runner.spec.prompt(query, documents[0])
    assert len(runner.tokenizer.encode(runner.spec.prompt(query, documents[0])).ids) <= 64
    with pytest.raises(ValueError, match="Query exceeds"):
        runner.prepare("question " * 100, ["document"])


def test_zerank_uses_its_published_roles_and_final_newline():
    spec = LOCAL_MODELS["zerank-2"]
    assert spec.prompt("question", "document") == (
        "<|im_start|>system\nquestion<|im_end|>\n<|im_start|>user\ndocument<|im_end|>\n<|im_start|>assistant\n"
    )
    assert spec.classifier == ("Yes",) and spec.method == "no_post_processing"


@pytest.mark.parametrize("flags", [("False", "False"), ("False", "True"), ("False", None)])
def test_local_timing_requires_verified_uncached_replicas(tokenizer_file, monkeypatch, flags):
    urls = ("http://first", "http://second")
    monkeypatch.setenv("ZERANK_RERANK_URLS", ",".join(urls))
    seen = []

    def respond(request):
        assert request.url.path == "/metrics" and request.method == "GET"
        seen.append(request.url.host)
        flag = flags[urls.index(f"http://{request.url.host}")]
        metrics = f'vllm:cache_config_info{{enable_prefix_caching="{flag}",engine="0"}} 1.0\n' if flag else ""
        return httpx.Response(200, text=metrics)

    runner = LocalReranker("zerank-2", client=httpx.Client(transport=httpx.MockTransport(respond)))
    if all(flag == "False" for flag in flags):
        assert runner.validate_latency() == {url: {"prefix_caching": False} for url in urls}
    else:
        with pytest.raises(ValueError, match="cannot verify prefix caching is disabled"):
            runner.validate_latency()
    assert seen == ["first", "second"]


def test_serving_disables_prefix_caching(monkeypatch):
    from types import SimpleNamespace

    from evals_repro import serve

    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(poll=lambda: 1, wait=lambda **kw: 1)

    monkeypatch.setattr(serve.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(serve.subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="server exited"):
        serve.serve(SimpleNamespace(model="zerank-2", devices=["3"], host="127.0.0.1", port=8400, memory_fraction=0.85))
    command, options = calls[0]
    assert "--no-enable-prefix-caching" in command
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "3"
