"""LSWMD.pkl -> checked Parquet -> lot yield and SPC.

    python scripts/run_pipeline.py [path/to/LSWMD.pkl]

Writes data/wafers.parquet, data/quarantine.parquet, results/lot_spc.parquet,
results/pipeline_report.json and results/p_chart.png.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wafer_yield.etl import process  # noqa: E402
from wafer_yield.source import records  # noqa: E402
from wafer_yield.spc import lot_table, p_chart, run_rule  # noqa: E402


def main(pkl: Path) -> None:
    data, results = ROOT / "data", ROOT / "results"
    results.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    stats = process(records(pkl), data)
    t_etl = time.perf_counter() - t0

    wafers = pd.read_parquet(data / "wafers.parquet", columns=[
        "lot", "wafer_index", "split", "label", "rows", "cols", "dies", "fails", "yield", "flags"])
    lots = lot_table(wafers)
    # The naive setup (one chart for every product) first, to show why it fails.
    one_plain, one_laney = p_chart(lots, laney=False, by=None), p_chart(lots, laney=True, by=None)
    plain = p_chart(lots, laney=False)
    chart = p_chart(lots, laney=True)
    chart["plain_out_of_control"] = plain["out_of_control"]
    chart["run_rule"] = run_rule(chart)
    chart.to_parquet(results / "lot_spc.parquet")
    charted = chart[chart["charted"]]

    labeled = wafers.dropna(subset=["label"])
    by_pattern = (labeled.groupby("label")["yield"].agg(["size", "mean", "median"])
                  .sort_values("mean").round(4))
    report = {
        "etl": stats, "etl_seconds": round(t_etl, 1),
        "split_counts": wafers["split"].fillna("unlabeled").value_counts().to_dict(),
        "label_counts": labeled["label"].value_counts().to_dict(),
        "distinct_map_shapes": int(wafers[["rows", "cols"]].drop_duplicates().shape[0]),
        "lots": int(len(chart)), "wafers_per_lot_median": float(chart["wafers"].median()),
        "overall_die_yield": round(1 - float(chart["fails"].sum() / chart["dies"].sum()), 4),
        "one_chart_all_products": {
            "p_chart_out_of_control_share": round(float(one_plain["out_of_control"].mean()), 4),
            "laney_sigma_z": round(float(one_laney["sigma_z"].iloc[0]), 2),
            "laney_out_of_control_share": round(float(one_laney["out_of_control"].mean()), 4),
            "run_rule_hits": int(run_rule(one_laney, by=None).sum()),
        },
        "per_product": {
            "products": int(lots["product"].nunique()),
            "products_charted": int(charted["product"].nunique()),
            "lots_charted": int(len(charted)),
            "wafers_in_charted_lots": int(charted["wafers"].sum()),
            "p_chart_out_of_control_share": round(float(plain.loc[plain["charted"], "out_of_control"].mean()), 4),
            "laney_sigma_z_median": round(float(charted.groupby("product")["sigma_z"].first().median()), 2),
            "laney_out_of_control_lots": int(charted["out_of_control"].sum()),
            "laney_out_of_control_share": round(float(charted["out_of_control"].mean()), 4),
            "run_rule_hits": int(chart["run_rule"].sum()),
            "out_of_control_top_patterns": charted.loc[charted["out_of_control"], "top_pattern"]
                                           .fillna("no labeled defect").value_counts().head(6).to_dict(),
        },
        "yield_by_pattern": by_pattern.reset_index().to_dict(orient="records"),
    }
    (results / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    plot(chart, results / "p_chart.png")
    print(json.dumps({k: v for k, v in report.items() if k != "yield_by_pattern"}, indent=2))
    print(by_pattern)


def plot(chart: pd.DataFrame, path: Path, n: int = 300) -> None:
    """The busiest product's chart: its first n lots."""
    top = chart.loc[chart["charted"], "product"].value_counts().index[0]
    c = chart[chart["product"] == top].head(n).reset_index(drop=True)
    x = range(len(c))
    sigma = (c["ucl"] - c["center"]) / c["sigma_z"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, c["fail_rate"], ".", ms=4, color="#333", label="lot fail rate")
    ax.plot(x, c["center"], color="#888", lw=1, label="centre")
    ax.plot(x, c["center"] + sigma, color="#d62728", lw=0.8, ls="--", label="p-chart UCL")
    ax.plot(x, c["ucl"], color="#1f77b4", lw=1.2, label="Laney p' UCL")
    hit = c["out_of_control"]
    ax.plot([i for i, h in zip(x, hit) if h], c.loc[hit, "fail_rate"], "o", mfc="none", color="#1f77b4", ms=8)
    ax.set_title(f"Product {top} (map size), {len(c)} lots")
    ax.set_xlabel(f"lot (first {n} of this product, by lot number)")
    ax.set_ylabel("failing die fraction")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "LSWMD.pkl")
