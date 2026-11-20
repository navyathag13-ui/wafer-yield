"""Picture for the README: one real Test wafer per defect pattern, with the ensemble's answer,
and one Scratch wafer next to its most similar Training wafers.

    python scripts/examples.py      (after run_pipeline.py and train.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

from wafer_yield.api import app  # noqa: E402
from wafer_yield.etl import decode  # noqa: E402
from wafer_yield.maps import CLASSES  # noqa: E402

CMAP = ListedColormap(["#ffffff", "#9ecae1", "#d62728"])  # no die, pass, fail


def main() -> None:
    client = TestClient(app)
    w = pd.read_parquet(ROOT / "data" / "wafers.parquet", filters=[("split", "==", "Test")]).dropna(subset=["label"])
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    for ax, label in zip(axes[0], CLASSES[:8]):
        pool = w[w["label"] == label]
        row = pool.iloc[int(rng.integers(len(pool)))]
        wafer = decode(row)
        out = client.post("/classify", json={"wafer_map": wafer.tolist()}).json()
        ax.imshow(wafer, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        ok = "ok" if out["pattern"] == label else "miss"
        ax.set_title(f"{label}\npred {out['pattern']} {out['confidence']:.2f} ({ok})", fontsize=8,
                     color="black" if ok == "ok" else "#d62728")
        ax.axis("off")
    scratch = w[w["label"] == "Scratch"].iloc[int(rng.integers((w["label"] == "Scratch").sum()))]
    hits = client.get(f"/similar/{scratch['lot']}/{int(scratch['wafer_index'])}?k=7").json()["similar"]
    all_w = pd.read_parquet(ROOT / "data" / "wafers.parquet", columns=["lot", "wafer_index", "rows", "cols", "map"])
    axes[1][0].imshow(decode(scratch), cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    axes[1][0].set_title(f"query: Scratch\n{scratch['lot']}", fontsize=8)
    for ax, h in zip(axes[1][1:], hits):
        r = all_w[(all_w["lot"] == h["lot"]) & (all_w["wafer_index"] == h["wafer_index"])].iloc[0]
        ax.imshow(decode(r), cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"similar: {h['label']}\nsim {h['similarity']:.3f}", fontsize=8)
    for ax in axes[1]:
        ax.axis("off")
    fig.suptitle("Top: one random Test wafer per pattern, ensemble answer.  Bottom: a Scratch wafer and its "
                 "7 most similar wafers from other lots.  Red = failed die.", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "examples.png", dpi=110)


if __name__ == "__main__":
    main()
