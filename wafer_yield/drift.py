"""Drift alarm: warn when a new batch of wafers stops looking like the wafers the model was
checked on.

The model's scores only hold for wafers like the ones it was validated on. On WM-811K's Test
split the review flag sent 10.7% of wafers to review instead of the 5% it was tuned for,
because the Test lots differ. In a fab that should raise an alarm before anyone trusts the
labels. Three numbers per batch, each compared with what normal batches look like:

  review_share  share of wafers the ensemble is unsure about
  mix_psi       population stability index of the predicted pattern mix against the reference
  novelty       median distance (1 - cosine similarity) from each wafer to its closest
                training wafer in the CNN embedding: new layouts and new defects sit far away

Thresholds are the 99th percentile of each number over batches drawn from reference
(validation) wafers. Batches are built from whole lots, because wafers arrive by lot and
lots differ from each other: batches of random single wafers hide that spread, and an alarm
calibrated on them fired on 61% of normal held-out batches. A batch alarms when any number
is above its threshold.
"""
from __future__ import annotations

import numpy as np

from .similar import normalize

METRICS = ("review_share", "mix_psi", "novelty")


def psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-3) -> float:
    """Population stability index between two distributions over the same categories."""
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    p, q = p / p.sum(), q / q.sum()
    return float(((p - q) * np.log(p / q)).sum())


def nearest_distance(vectors: np.ndarray, base: np.ndarray, batch: int = 2048) -> np.ndarray:
    """1 - cosine similarity to the closest base vector, per row."""
    base = normalize(base)
    out = []
    for i in range(0, len(vectors), batch):
        out.append(1 - (normalize(vectors[i:i + batch]) @ base.T).max(1))
    return np.concatenate(out) if out else np.zeros(0)


def lot_batch(lots: np.ndarray | None, size: int, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Row indices for one batch of about `size` wafers made of whole lots drawn at random
    (with replacement). Without lots, plain random rows from range(n)."""
    if lots is None:
        return rng.integers(0, n, size)
    uniq, inverse = np.unique(lots, return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    rows: list[int] = []
    while len(rows) < size:
        rows.extend(members[int(rng.integers(len(uniq)))].tolist())
    return np.asarray(rows[:size])


class DriftMonitor:
    def __init__(self, ref_probs: np.ndarray, ref_distance: np.ndarray, review_threshold: float,
                 ref_lots: np.ndarray | None = None, batch: int = 500, n_boot: int = 400,
                 quantile: float = 0.99, seed: int = 0):
        self.review_threshold = review_threshold
        self.n_classes = ref_probs.shape[1]
        self.ref_mix = np.bincount(ref_probs.argmax(1), minlength=self.n_classes) / len(ref_probs)
        rng = np.random.default_rng(seed)
        boots = [self.measure(ref_probs[i], ref_distance[i])
                 for i in (lot_batch(ref_lots, batch, rng, n=len(ref_probs)) for _ in range(n_boot))]
        self.thresholds = {m: float(np.quantile([b[m] for b in boots], quantile)) for m in METRICS}

    def measure(self, probs: np.ndarray, distance: np.ndarray) -> dict:
        mix = np.bincount(probs.argmax(1), minlength=self.n_classes) / max(len(probs), 1)
        return {"review_share": float((probs.max(1) < self.review_threshold).mean()),
                "mix_psi": psi(mix, self.ref_mix),
                "novelty": float(np.median(distance))}

    def check(self, probs: np.ndarray, distance: np.ndarray) -> dict:
        m = self.measure(probs, distance)
        fired = [k for k in METRICS if m[k] > self.thresholds[k]]
        return {**{k: round(v, 4) for k, v in m.items()}, "alarm": bool(fired), "fired": fired}
