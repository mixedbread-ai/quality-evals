import pytest

from evals_repro.cli import store_retriever


def test_store_method_names_parse(monkeypatch):
    monkeypatch.setenv("MXBAI_API_KEY", "test")
    assert store_retriever("mixedbread-images+rerank", "p", 10).name == "mixedbread-images+mxbai-rerank-v3.1-listwise"
    assert store_retriever("mixedbread-markdown", "p", 10).rerank_top_k is None


def test_summarize_empty():
    from evals_repro.metrics import summarize

    assert summarize({}) == {}


@pytest.mark.parametrize(
    "argv", [["run", "bm25", "--content", "images"], ["run", "mixedbread-markdown", "--rerank", "cohere"]]
)
def test_cli_rejects_incoherent_combinations(argv, monkeypatch):
    import sys

    from evals_repro import cli

    monkeypatch.setattr(sys, "argv", ["evals-repro", *argv])
    with pytest.raises(SystemExit):
        cli.main()
