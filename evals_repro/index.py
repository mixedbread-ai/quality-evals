from collections.abc import Sequence

import numpy as np

Run = dict[str, dict[str, float]]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class ExactIndex:
    def __init__(self, ids: Sequence[str], vectors: np.ndarray):
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} ids for {len(vectors)} vectors")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids")
        self.ids = np.asarray(ids, dtype=object)
        self.vectors = l2_normalize(vectors)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, queries: np.ndarray, k: int, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        queries = l2_normalize(np.atleast_2d(queries))
        k = min(k, len(self))
        ids, scores = [], []
        for start in range(0, len(queries), batch_size):
            batch_scores = queries[start : start + batch_size] @ self.vectors.T
            top = np.argsort(-batch_scores, axis=1, kind="stable")[:, :k]
            ids.append(self.ids[top])
            scores.append(np.take_along_axis(batch_scores, top, axis=1))
        return np.concatenate(ids), np.concatenate(scores)

    def run(self, query_ids: Sequence[str], queries: np.ndarray, k: int) -> Run:
        ids, scores = self.search(queries, k)
        return {
            qid: dict(zip(row_ids.tolist(), row_scores.tolist(), strict=True))
            for qid, row_ids, row_scores in zip(query_ids, ids, scores, strict=True)
        }
