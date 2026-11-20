"""Find past wafers that look like a new one.

When a wafer fails in a new way, an engineer's first question is "have we seen this before,
and what was the cause?". Each wafer is turned into 64 numbers by the CNN (cnn.WaferCNN.embed),
and the closest past wafers by cosine similarity are returned with their lot and label.
Plain numpy: 172,950 x 64 fits in memory and one query is a single matrix-vector product.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class SimilarIndex:
    def __init__(self, vectors: np.ndarray, meta: pd.DataFrame):
        if len(vectors) != len(meta):
            raise ValueError("one row of metadata per vector")
        self.vectors = normalize(vectors)
        self.meta = meta.reset_index(drop=True)

    @classmethod
    def load(cls, folder: Path) -> "SimilarIndex":
        return cls(np.load(folder / "embeddings.npy"), pd.read_parquet(folder / "embedding_index.parquet"))

    def query(self, vector: np.ndarray, k: int = 10, exclude_lot: str | None = None,
              mask: np.ndarray | None = None) -> pd.DataFrame:
        """Top k rows by cosine similarity. exclude_lot drops the query's own lot (its sister
        wafers would otherwise fill the list); mask limits the search to some rows."""
        sims = self.vectors @ normalize(vector)
        ok = np.ones(len(sims), dtype=bool) if mask is None else mask.copy()
        if exclude_lot is not None:
            ok &= self.meta["lot"].to_numpy() != exclude_lot
        sims = np.where(ok, sims, -np.inf)
        k = min(k, int(ok.sum()))
        top = np.argpartition(-sims, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-sims[top])]
        return self.meta.iloc[top].assign(similarity=sims[top].round(4))


def precision_at_k(query_vecs, query_labels, base_vecs, base_labels, k: int = 10, batch: int = 2048) -> np.ndarray:
    """For each query, the share of its k nearest base rows that carry the same label."""
    base = normalize(base_vecs)
    base_labels = np.asarray(base_labels)
    out = []
    for i in range(0, len(query_vecs), batch):
        sims = normalize(query_vecs[i:i + batch]) @ base.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out.append((base_labels[top] == np.asarray(query_labels[i:i + batch])[:, None]).mean(1))
    return np.concatenate(out) if out else np.zeros(0)
