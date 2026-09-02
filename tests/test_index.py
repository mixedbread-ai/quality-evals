import numpy as np
import pytest

from evals_repro.index import ExactIndex, l2_normalize


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def same_order_up_to_ties(got, want, want_scores, tol=1e-5):
    if got == want:
        return True
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if g != w and not any(abs(want_scores[i] - want_scores[j]) < tol for j in range(len(want)) if want[j] == g):
            return False
    return True


def naive_search(ids, docs, query, k):
    ranked = sorted(range(len(ids)), key=lambda i: (-cosine(query, docs[i]), i))[:k]
    return [ids[i] for i in ranked], [cosine(query, docs[i]) for i in ranked]


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(500, 32)) * rng.uniform(0.1, 10, size=(500, 1))
    return [f"d{i}" for i in range(500)], docs


@pytest.mark.parametrize("k", [1, 7, 100, 500, 1000])
def test_matches_naive_search(corpus, k):
    ids, docs = corpus
    index = ExactIndex(ids, docs)
    queries = np.random.default_rng(1).normal(size=(40, 32))
    got_ids, got_scores = index.search(queries, k)
    assert got_ids.shape == got_scores.shape == (40, min(k, 500))
    for query, row_ids, row_scores in zip(queries, got_ids, got_scores, strict=True):
        want_ids, want_scores = naive_search(ids, docs, query, k)
        assert np.allclose(row_scores, want_scores, atol=1e-5)
        assert same_order_up_to_ties(row_ids.tolist(), want_ids, want_scores)
        assert np.allclose([cosine(query, docs[int(i[1:])]) for i in row_ids], row_scores, atol=1e-5)


def test_batching_does_not_change_results(corpus):
    ids, docs = corpus
    index = ExactIndex(ids, docs)
    queries = np.random.default_rng(2).normal(size=(37, 32))
    whole = index.search(queries, 10, batch_size=1000)
    pieces = index.search(queries, 10, batch_size=5)
    assert np.allclose(whole[1], pieces[1], atol=1e-6)
    for whole_ids, piece_ids, scores in zip(whole[0], pieces[0], whole[1], strict=True):
        assert same_order_up_to_ties(piece_ids.tolist(), whole_ids.tolist(), scores.tolist(), tol=1e-6)


def test_planted_neighbour_is_found(corpus):
    ids, docs = corpus
    index = ExactIndex(ids, docs)
    targets = [3, 250, 499]
    queries = docs[targets] * 3 + np.random.default_rng(3).normal(scale=1e-3, size=(3, 32))
    got_ids, got_scores = index.search(queries, 1)
    assert got_ids[:, 0].tolist() == [ids[t] for t in targets]
    assert np.all(got_scores[:, 0] > 0.999)


def test_ties_break_by_insertion_order():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    got_ids, got_scores = ExactIndex(["a", "b", "c", "d"], docs).search(np.array([[1.0, 0.0]]), 4)
    assert got_ids[0].tolist() == ["a", "c", "d", "b"]
    assert np.allclose(got_scores[0], [1, 1, 1, 0])


def test_scale_invariance(corpus):
    ids, docs = corpus
    query = np.random.default_rng(4).normal(size=32)
    scaled = ExactIndex(ids, docs * 1e3).search(query * 1e-3, 20)
    plain = ExactIndex(ids, docs).search(query, 20)
    assert scaled[0].tolist() == plain[0].tolist()
    assert np.allclose(scaled[1], plain[1], atol=1e-5)


def test_single_query_and_low_precision_input(corpus):
    ids, docs = corpus
    index = ExactIndex(ids, docs.astype(np.float16))
    got_ids, got_scores = index.search(docs[10].astype(np.float16), 3)
    assert got_ids.shape == (1, 3) and got_ids[0, 0] == "d10"


def test_run_uses_query_ids(corpus):
    ids, docs = corpus
    run = ExactIndex(ids, docs).run(["q1", "q2"], docs[[5, 6]], 3)
    assert list(run) == ["q1", "q2"]
    assert max(run["q1"], key=run["q1"].get) == "d5"
    assert len(run["q2"]) == 3


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        ExactIndex(["a", "b"], np.ones((3, 2)))
    with pytest.raises(ValueError):
        ExactIndex(["a", "a"], np.ones((2, 2)))


def test_zero_vectors_do_not_produce_nan():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert not np.isnan(l2_normalize(vectors)).any()
    got_ids, got_scores = ExactIndex(["z", "u"], vectors).search(np.zeros((1, 2)), 2)
    assert not np.isnan(got_scores).any()
