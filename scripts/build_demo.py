"""Build the static demo in docs/ (GitHub Pages serves it; no server, no model in the browser).

    python scripts/build_demo.py      (after run_pipeline.py, train.py and audit.py)

Writes docs/demo.json (sample Test wafers with the ensemble's answers, their most similar
wafers, label-audit suspects, lot alarms, drift and headline numbers) and copies the charts
to docs/img/. docs/index.html reads demo.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wafer_yield.defects import GUIDE, SOURCES  # noqa: E402
from wafer_yield.etl import decode  # noqa: E402
from wafer_yield.maps import CLASSES  # noqa: E402
from wafer_yield.similar import SimilarIndex  # noqa: E402

RES, DOCS = ROOT / "results", ROOT / "docs"
PER_DEFECT, N_NONE, K = 14, 30, 5


def encode(wafer: np.ndarray) -> dict:
    return {"r": int(wafer.shape[0]), "c": int(wafer.shape[1]), "d": "".join(map(str, wafer.ravel().tolist()))}


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "img").mkdir(exist_ok=True)
    metrics = json.loads((RES / "metrics.json").read_text())
    audit = json.loads((RES / "audit.json").read_text())
    pipe = json.loads((RES / "pipeline_report.json").read_text())
    cfg = json.loads((RES / "ensemble.json").read_text())
    pred = pd.read_parquet(RES / "labeled_predictions.parquet")
    idx = SimilarIndex.load(RES)
    maps = pd.read_parquet(ROOT / "data" / "wafers.parquet", columns=["lot", "wafer_index", "rows", "cols", "map"])
    maps = maps.set_index(["lot", "wafer_index"])

    def wafer_map(lot, i):
        return encode(decode(maps.loc[(lot, int(i))]))

    probs = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    test = pred[pred["role"] == "test"]
    rng = np.random.default_rng(7)
    picks = []
    for c in CLASSES:
        rows = test.index[test["label"] == c].to_numpy()
        picks += rng.choice(rows, min(N_NONE if c == "none" else PER_DEFECT, len(rows)), replace=False).tolist()
    rng.shuffle(picks)
    train_mask = (idx.meta["split"] == "Training").to_numpy()
    wafers = []
    for i in picks:
        r = pred.loc[i]
        p = probs[i]
        order = np.argsort(p)[::-1]
        sim = idx.query(idx.vectors[i], k=K, exclude_lot=r["lot"], mask=train_mask)
        wafers.append({
            "lot": r["lot"], "i": int(r["wafer_index"]), "label": r["label"], "map": wafer_map(r["lot"], r["wafer_index"]),
            "pred": CLASSES[order[0]], "conf": round(float(p[order[0]]), 3),
            "review": bool(p[order[0]] < cfg["review_threshold"]),
            "top3": [[CLASSES[j], round(float(p[j]), 3)] for j in order[:3]],
            "similar": [{"lot": s.lot, "i": int(s.wafer_index), "label": s.label, "sim": float(s.similarity),
                         "map": wafer_map(s.lot, s.wafer_index)} for s in sim.itertuples()],
        })
    sus = pd.read_parquet(RES / "label_audit.parquet")
    sus_sample = sus.sample(min(16, len(sus)), random_state=3)
    suspects = [{"lot": s.lot, "i": int(s.wafer_index), "label": s.label, "looks": s.vote_label,
                 "votes": int(s.vote_count), "p": round(float(s.p_vote), 3), "map": wafer_map(s.lot, s.wafer_index)}
                for s in sus_sample.itertuples()]
    spc = pd.read_parquet(RES / "lot_spc.parquet")
    # Flagged lots that also hold labeled defect wafers: that is where SPC meets the pattern classifier.
    flagged = spc[spc["out_of_control"]]
    worst = (flagged[flagged["top_pattern"].notna()].assign(excess=lambda d: d["fail_rate"] - d["ucl"])
             .sort_values("excess", ascending=False).head(12))
    lots = [{"lot": r.lot, "product": r.product, "wafers": int(r.wafers), "fail_rate": round(float(r.fail_rate), 3),
             "ucl": round(float(r.ucl), 3), "pattern": None if pd.isna(r.top_pattern) else r.top_pattern}
            for r in worst.itertuples()]
    data = {
        "classes": CLASSES, "review_threshold": cfg["review_threshold"],
        "guide": GUIDE, "sources": {k: {"text": v[0], "url": v[1]} for k, v in SOURCES.items()},
        "wafers": wafers, "suspects": suspects, "lots": lots,
        "flagged_with_pattern": int(flagged["top_pattern"].notna().sum()), "flagged_total": int(len(flagged)),
        "numbers": {"wafers": pipe["etl"]["input"], "lots": pipe["lots"], "labeled": pipe["etl"]["labeled"],
                    "test": metrics["test"], "ensemble": metrics["ensemble"], "cnn": metrics["cnn"],
                    "trees": metrics["baseline_gbt"], "review": metrics["review"][:-1],
                    "review_chosen": metrics["review"][-1],
                    "similar": metrics["similar_wafers"], "audit": audit["label_audit"], "drift": audit["drift"],
                    "spc": pipe["per_product"], "spc_one_chart": pipe["one_chart_all_products"]},
    }
    (DOCS / "demo.json").write_text(json.dumps(data, separators=(",", ":")))
    for name in ("p_chart.png", "drift.png", "confusion_ensemble.png"):
        shutil.copy(RES / name, DOCS / "img" / name)
    print(f"docs/demo.json: {len(wafers)} wafers, {len(suspects)} suspects, {len(lots)} lots, "
          f"{(DOCS / 'demo.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
