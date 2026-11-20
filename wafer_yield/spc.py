"""Lot yield and statistical process control (SPC).

A p-chart watches the fraction of failing dies per lot. Its limits are
p_bar +/- 3 * sqrt(p_bar * (1 - p_bar) / n), where n is the dies in that lot.

Two things go wrong on real fab data, and both showed up on WM-811K:

1. Products differ. Each product (here: each wafer map size, i.e. die layout) has its own
   normal fail rate, so one chart across all products mostly measures product mix. Charts are
   built per product; products with fewer than MIN_LOTS lots are left uncharted.
2. Lots vary more than the binomial model expects. A lot has thousands of dies, so plain
   limits are very tight and normal lot-to-lot variation lands outside them. Laney's p' chart
   measures that extra spread (sigma_z, from the moving range of z-scores) and widens the
   limits by it.

The lots carry names, not timestamps, so "order" is lot number within a product. The run rule
is a demonstration of the method, not proof of a drift in time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LOTS = 20


def lot_table(wafers: pd.DataFrame) -> pd.DataFrame:
    """One row per lot: product, wafers, dies, fails, die yield and most common defect pattern."""
    w = wafers.copy()
    if {"rows", "cols"} <= set(w.columns):
        w["product"] = w["rows"].astype(str) + "x" + w["cols"].astype(str)
    else:
        w["product"] = "all"
    g = w.groupby("lot", sort=False)
    lots = g.agg(wafers=("wafer_index", "size"), dies=("dies", "sum"), fails=("fails", "sum"),
                 product=("product", lambda s: s.mode().iloc[0]))
    lots["fail_rate"] = lots["fails"] / lots["dies"]
    lots["yield"] = 1 - lots["fail_rate"]
    if "label" in w:
        top = (w.dropna(subset=["label"]).query("label != 'none'")
               .groupby("lot")["label"].agg(lambda s: s.value_counts().index[0]))
        lots["top_pattern"] = top
    lots = lots.reset_index()
    lots["lot_number"] = pd.to_numeric(lots["lot"].str.extract(r"(\d+)")[0], errors="coerce")
    return lots.sort_values("lot_number", kind="stable").reset_index(drop=True)


def _limits(group: pd.DataFrame, laney: bool) -> pd.DataFrame:
    p_bar = group["fails"].sum() / group["dies"].sum()
    sigma = np.sqrt(p_bar * (1 - p_bar) / group["dies"])
    sigma_z = 1.0
    if laney and len(group) > 1:
        z = (group["fail_rate"] - p_bar) / sigma.replace(0, np.nan)
        # Spread of z from the average moving range, as in an individuals chart (d2 = 1.128).
        sigma_z = float(np.nanmean(np.abs(np.diff(z))) / 1.128) or 1.0
    return group.assign(center=p_bar, sigma_z=sigma_z, ucl=(p_bar + 3 * sigma * sigma_z).clip(upper=1),
                        lcl=(p_bar - 3 * sigma * sigma_z).clip(lower=0))


def p_chart(lots: pd.DataFrame, laney: bool = True, by: str | None = "product",
            min_lots: int = MIN_LOTS) -> pd.DataFrame:
    """Add centre line, limits and an out-of-control flag per product (or overall when by=None).

    Lots of a product with fewer than min_lots lots get no limits and charted=False."""
    out = lots.copy()
    key = out[by] if by else pd.Series("all", index=out.index)
    sizes = key.map(key.value_counts())
    out["charted"] = sizes >= min_lots if by else True
    parts = [_limits(g, laney) for _, g in out[out["charted"]].groupby(key[out["charted"]], sort=False)]
    if parts:
        lim = pd.concat(parts)[["center", "sigma_z", "ucl", "lcl"]]
        out = out.join(lim)
    else:
        out[["center", "sigma_z", "ucl", "lcl"]] = np.nan
    out["out_of_control"] = out["charted"] & ((out["fail_rate"] > out["ucl"]) | (out["fail_rate"] < out["lcl"]))
    return out


def run_rule(chart: pd.DataFrame, length: int = 8, by: str | None = "product") -> pd.Series:
    """Western Electric rule 4: `length` lots in a row (within a product) on one side of the centre."""
    hits = pd.Series(False, index=chart.index)
    groups = chart.groupby(by, sort=False) if by and by in chart else [("all", chart)]
    for _, g in groups:
        if "charted" in g and not g["charted"].all():
            continue
        side = np.sign(g["fail_rate"] - g["center"]).to_numpy()
        run = 0
        for i, label in enumerate(g.index):
            run = run + 1 if i and side[i] == side[i - 1] and side[i] != 0 else 1
            if run >= length:
                hits[label] = True
    return hits
