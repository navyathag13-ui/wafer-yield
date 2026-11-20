import numpy as np
import pandas as pd

from wafer_yield.similar import SimilarIndex, precision_at_k


def test_query_ranks_by_cosine_and_excludes_lot():
    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    meta = pd.DataFrame({"lot": ["a", "b", "c", "d"], "label": ["x", "x", "y", "z"]})
    idx = SimilarIndex(vecs, meta)
    top = idx.query(np.array([1, 0]), k=2)
    assert list(top["lot"]) == ["a", "b"] and top["similarity"].iloc[0] == 1.0
    assert list(idx.query(np.array([1, 0]), k=2, exclude_lot="a")["lot"]) == ["b", "c"]
    assert len(idx.query(np.array([1, 0]), k=10)) == 4


def test_precision_at_k():
    base = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]])
    p = precision_at_k(np.array([[1, 0], [0, 1]]), np.array([0, 1]), base, np.array([0, 0, 1, 1]), k=2)
    assert np.allclose(p, [1, 1])
    p = precision_at_k(np.array([[1, 0]]), np.array([1]), base, np.array([0, 0, 1, 1]), k=2)
    assert np.allclose(p, [0])
