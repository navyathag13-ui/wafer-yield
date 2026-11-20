"""Hand-made features for the baseline model. Each one matches how an engineer reads a map:
where the fails sit (centre, ring, edge, one side), how many there are, and whether they
form one long cluster (a scratch) or one round blob (a local defect)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .maps import FAIL, OFF, SIZE, resize

RINGS = [(0.0, 0.3), (0.3, 0.55), (0.55, 0.8), (0.8, 0.92), (0.92, 1.01)]
NAMES = (["fail_ratio"] + [f"ring_{a}_{b}" for a, b in RINGS] + [f"sector_{k}" for k in range(8)]
         + ["sector_spread", "n_clusters", "top_cluster_share", "top_cluster_radius",
            "top_cluster_elongation", "top_cluster_fill"])

_yy, _xx = np.mgrid[:SIZE, :SIZE]
_c = (SIZE - 1) / 2
_R = np.hypot(_yy - _c, _xx - _c) / (SIZE / 2)
_SECTOR = ((np.arctan2(_yy - _c, _xx - _c) + np.pi) / (2 * np.pi) * 8).astype(int) % 8


def features(wafer: np.ndarray) -> np.ndarray:
    m = resize(wafer)
    on, fail = m > OFF, m == FAIL
    n_on = max(on.sum(), 1)
    # Radius from the wafer's own outline, so oval or off-centre grids still give 1.0 at the edge.
    r = _R / max(_R[on].max(), 1e-6)
    out = [fail.sum() / n_on]
    for a, b in RINGS:
        zone = on & (r >= a) & (r < b)
        out.append(fail[zone].mean() if zone.any() else 0.0)
    sectors = [fail[on & (_SECTOR == k)].mean() if (on & (_SECTOR == k)).any() else 0.0 for k in range(8)]
    out += sectors + [float(np.std(sectors))]
    # 8-connected: scratches are often diagonal, and 4-connectivity breaks them into single dies.
    labels, n = ndimage.label(fail, structure=np.ones((3, 3), dtype=int))
    if n:
        sizes = ndimage.sum(fail, labels, range(1, n + 1))
        top = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(labels == top)
        cy, cx = ys.mean(), xs.mean()
        if len(ys) > 2:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs])))
            elong = float(np.sqrt(ev[-1] / max(ev[0], 1e-3)))
        else:
            elong = 1.0
        box = (np.ptp(ys) + 1) * (np.ptp(xs) + 1)
        out += [n, sizes.max() / max(fail.sum(), 1), float(r[int(round(cy)), int(round(cx))]), elong, len(ys) / box]
    else:
        out += [0, 0.0, 0.0, 0.0, 0.0]
    return np.asarray(out, dtype=np.float32)
