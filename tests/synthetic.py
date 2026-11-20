"""Small fake wafers with known patterns, so tests never need the 2 GB dataset."""
from __future__ import annotations

import numpy as np

from wafer_yield.maps import FAIL, OFF, PASS


def blank(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """A round wafer of passing dies, plus each cell's radius (0 at centre, 1 at the edge)."""
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c) / (n / 2)
    wafer = np.where(r <= 1.0, PASS, OFF).astype(np.uint8)
    return wafer, r


def make(kind: str, n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, r = blank(n)
    on = w > OFF
    fail = np.zeros_like(on)
    if kind == "Center":
        fail = r < 0.3
    elif kind == "Edge-Ring":
        fail = r > 0.85
    elif kind == "Donut":
        fail = (r > 0.35) & (r < 0.6)
    elif kind == "Scratch":
        yy, xx = np.mgrid[:n, :n]
        fail = (np.abs(yy - xx) <= 0) & (r < 0.8)
    elif kind == "Loc":
        yy, xx = np.mgrid[:n, :n]
        fail = np.hypot(yy - n * 0.3, xx - n * 0.6) < n * 0.1
    elif kind == "Edge-Loc":
        yy, xx = np.mgrid[:n, :n]
        fail = (r > 0.75) & (np.arctan2(yy - n / 2, xx - n / 2) > 0.4) & (np.arctan2(yy - n / 2, xx - n / 2) < 1.4)
    elif kind == "Random":
        fail = rng.random(w.shape) < 0.2
    elif kind == "Near-full":
        fail = rng.random(w.shape) < 0.9
    elif kind != "none":
        raise ValueError(kind)
    fail = fail | (rng.random(w.shape) < 0.01)  # every real wafer has a few random fails
    w[on & fail] = FAIL
    return w
