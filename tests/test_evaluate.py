import json

import numpy as np

from evals_repro.benchmarks import Benchmark
from evals_repro.data import Query, Subset
from evals_repro.evaluate import evaluate_subset
from evals_repro.report import report
from evals_repro.retrievers import DenseRetriever

LANGUAGES = ("english", "french", "german")
BENCHMARK = Benchmark("toy", ("computer_science",), LANGUAGES, lambda name: toy_subset())


class LookupEmbedder:
    name = "lookup"
    table = {"alpha": [1, 0], "beta": [0, 1], "": [0, 0], "find alpha": [0.9, 0.1], "find beta": [0.1, 0.9]}

    def embed(self, texts, kind):
        return np.array([self.table[t] for t in texts], dtype=np.float32)

    def embed_images(self, images):
        return np.array([[len(png), 1.0] for _, png in images], dtype=np.float32)


def images():
    return iter([("1", b"ab"), ("0", b"abcdefgh"), ("2", b"")])


def toy_subset():
    queries = [Query("q0", "find alpha", "english"), Query("q1", "find beta", "french")]
    pages = {"0": "alpha", "1": "beta", "2": ""}
    return Subset(
        "computer_science", "src", "english", pages, queries, {"q0": {"0": 2}, "q1": {"1": 1, "0": 2}}, images
    )


def test_end_to_end_with_toy_retriever(tmp_path):
    result = evaluate_subset(DenseRetriever(LookupEmbedder(), tmp_path), toy_subset(), ["english", "french"], top_k=3)
    english, french = result.languages
    assert english.metrics["ndcg_cut_10"] == 1.0 and english.num_queries == 1
    assert french.metrics["recall_10"] == 1.0 and french.per_query["q1"]["ndcg_cut_10"] < 1.0
    saved = json.loads(result.save(tmp_path / "results").read_text())
    assert saved["method"] == "lookup" and saved["native_language"] == "english"


def full_language_subset():
    subset = toy_subset()
    subset.queries += [Query(f"q{i}", "find alpha", lang) for i, lang in enumerate(LANGUAGES, start=2)]
    subset.qrels.update({f"q{i}": {"0": 1} for i in range(2, 5)})
    return subset


def test_report_tables(tmp_path):
    evaluate_subset(DenseRetriever(LookupEmbedder(), tmp_path), full_language_subset(), list(LANGUAGES), 3).save(
        tmp_path / "r"
    )
    text = report(tmp_path / "r", BENCHMARK)
    assert "| monolingual ndcg_cut_10 | computer_science | avg |" in text
    assert "| lookup | 100.00 | 100.00 |" in text
    assert "| crosslingual ndcg_cut_10 | computer_science | avg |" in text


def test_report_can_hide_reranked_methods_and_partial_languages(tmp_path):
    retriever = DenseRetriever(LookupEmbedder(), tmp_path)
    evaluate_subset(retriever, toy_subset(), ["english"], 3).save(tmp_path / "r")
    retriever.embedder.name = "lookup+rr@3"
    evaluate_subset(retriever, toy_subset(), ["english"], 3).save(tmp_path / "r")
    full = report(tmp_path / "r", BENCHMARK)
    assert "lookup+rr@3" in full and "lookup+rr@3" not in report(tmp_path / "r", BENCHMARK, first_stage_only=True)
    assert full.count("| lookup |") == 1


def test_image_content_indexes_page_images(tmp_path):
    retriever = DenseRetriever(LookupEmbedder(), tmp_path, content="images")
    run = retriever.run(toy_subset(), [Query("q", "find alpha", "english")], 2)
    assert retriever.name == "lookup-images" and max(run["q"], key=run["q"].get) == "0"
