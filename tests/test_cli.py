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


def test_study_defaults_and_all_benchmark_dispatch(monkeypatch):
    import sys

    from evals_repro import cli, study

    calls = []
    monkeypatch.setattr(study, "study", lambda args, benchmark: calls.append((args, benchmark.name)))
    monkeypatch.setattr(sys, "argv", ["evals-repro", "rerank-study", "latency"])
    cli.main()
    assert [name for _, name in calls] == list(study.BENCHMARKS)
    args = calls[0][0]
    assert (args.samples, args.repeats, args.depth, args.first_stage) == (20, 5, 100, "bm25")
    assert args.models == ["mixedbread", "voyage", "cohere-pro", "cohere-fast"]
    assert calls[-1][0].first_stage == "mixedbread-markdown"
    assert all(args.store_prefix == "vidore-eval" for args, _ in calls)


def test_study_bm25_override_and_custom_store_prefix(monkeypatch):
    import sys

    from evals_repro import cli, study

    calls = []
    monkeypatch.setattr(study, "study", lambda args, benchmark: calls.append(args))
    monkeypatch.setattr(
        sys, "argv", ["evals-repro", "rerank-study", "prepare", "--first-stage", "bm25", "--store-prefix", "custom"]
    )
    cli.main()
    assert len(calls) == 7 and all(args.first_stage == "bm25" and args.store_prefix == "custom" for args in calls)


def test_retrieval_store_prefix_does_not_inherit_study_defaults(monkeypatch):
    import sys

    from evals_repro import cli

    calls = []
    monkeypatch.setattr(cli, "upload", calls.append)
    monkeypatch.setattr(sys, "argv", ["evals-repro", "mixedbread-upload", "markdown"])
    cli.main()
    assert calls[0].store_prefix is None


@pytest.mark.parametrize(
    "options",
    [
        ["--samples", "0"],
        ["--repeats", "3"],
        ["--subsets", "fiqa"],
        ["--first-stage", "mixedbread-markdown"],
        ["--benchmark", "followir", "--depth", "50"],
        ["--models", "mixedbread", "mixedbread"],
        ["--benchmark", "text", "--subsets", "fiqa", "fiqa"],
    ],
)
def test_study_rejects_invalid_protocol(options, monkeypatch):
    import sys

    from evals_repro import cli

    monkeypatch.setattr(sys, "argv", ["evals-repro", "rerank-study", "quality", *options])
    with pytest.raises(SystemExit):
        cli.main()
