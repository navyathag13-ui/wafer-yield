"""FastAPI service over the pipeline's outputs.

  POST /classify            a wafer map (2D list of 0/1/2) -> pattern, confidence, needs_review, yield
  POST /similar             a wafer map -> the k most similar labeled wafers from history
  GET  /similar/{lot}/{i}   the same, for a wafer already in the data (its own lot is excluded)
  GET  /lots/{lot}          one lot's yield and SPC status
  GET  /spc/signals         lots outside the Laney p' limits, worst first

Run:  uvicorn wafer_yield.api:app --reload    (after scripts/run_pipeline.py and scripts/train.py)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import cnn
from .etl import check
from .features import features
from .maps import CLASSES, die_yield, to_channels
from .similar import SimilarIndex

ROOT = Path(os.environ.get("WAFER_ROOT", Path(__file__).resolve().parents[1]))
app = FastAPI(title="Wafer yield")


class WaferIn(BaseModel):
    wafer_map: list[list[int]] = Field(..., description="rows of die codes: 0 no die, 1 pass, 2 fail")


class SimilarIn(WaferIn):
    k: int = Field(10, ge=1, le=100)


@lru_cache
def model():
    path = ROOT / "results" / "cnn.pt"
    if not path.exists():
        raise HTTPException(503, "no trained model yet: run scripts/train.py")
    return cnn.load(path)


@lru_cache
def ensemble():
    """The tree model and ensemble settings, if scripts/train.py has saved them (else CNN alone)."""
    gbt_path, cfg_path = ROOT / "results" / "gbt.joblib", ROOT / "results" / "ensemble.json"
    if not (gbt_path.exists() and cfg_path.exists()):
        return None, {"cnn_weight": 1.0, "review_threshold": 0.0}
    return joblib.load(gbt_path), json.loads(cfg_path.read_text())


@lru_cache
def index() -> SimilarIndex:
    if not (ROOT / "results" / "embeddings.npy").exists():
        raise HTTPException(503, "no embeddings yet: run scripts/train.py")
    return SimilarIndex.load(ROOT / "results")


def parse(wafer_map: list[list[int]]) -> np.ndarray:
    if len({len(r) for r in wafer_map}) > 1:
        raise HTTPException(422, "rows have different lengths")
    wafer, reason = check(np.asarray(wafer_map))
    if wafer is None:
        raise HTTPException(422, reason)
    return wafer


def similar_rows(vector: np.ndarray, k: int, exclude_lot: str | None = None) -> list[dict]:
    rows = index().query(vector, k=k, exclude_lot=exclude_lot)
    return [{"lot": r.lot, "wafer_index": int(r.wafer_index), "label": r.label, "split": r.split,
             "similarity": float(r.similarity)} for r in rows.itertuples()]


@lru_cache
def chart() -> pd.DataFrame:
    path = ROOT / "results" / "lot_spc.parquet"
    if not path.exists():
        raise HTTPException(503, "no SPC table yet: run scripts/run_pipeline.py")
    return pd.read_parquet(path).set_index("lot")


@app.post("/classify")
def classify(body: WaferIn):
    wafer = parse(body.wafer_map)
    probs = cnn.predict(model(), [wafer])[0]
    gbt, cfg = ensemble()
    if gbt is not None:
        w = cfg["cnn_weight"]
        probs = w * probs + (1 - w) * gbt.predict_proba(features(wafer)[None])[0]
    order = np.argsort(probs)[::-1]
    conf = float(probs[order[0]])
    return {"pattern": CLASSES[order[0]], "confidence": round(conf, 4),
            "needs_review": conf < cfg["review_threshold"],
            "top3": [{"pattern": CLASSES[i], "p": round(float(probs[i]), 4)} for i in order[:3]],
            "die_yield": round(die_yield(wafer), 4)}


@app.get("/lots/{lot}")
def lot(lot: str):
    c = chart()
    if lot not in c.index:
        raise HTTPException(404, f"unknown lot {lot}")
    row = c.loc[lot]
    return {"lot": lot, "wafers": int(row["wafers"]), "yield": round(float(row["yield"]), 4),
            "fail_rate": round(float(row["fail_rate"]), 4), "ucl": round(float(row["ucl"]), 4),
            "lcl": round(float(row["lcl"]), 4), "out_of_control": bool(row["out_of_control"]),
            "top_pattern": None if pd.isna(row.get("top_pattern")) else row.get("top_pattern")}


@app.get("/spc/signals")
def signals(limit: int = 20):
    c = chart()
    hit = c[c["out_of_control"]].assign(excess=lambda d: d["fail_rate"] - d["ucl"])
    hit = hit.sort_values("excess", ascending=False).head(limit)
    return {"lots_checked": len(c), "out_of_control": int(c["out_of_control"].sum()),
            "worst": [{"lot": k, "fail_rate": round(float(v["fail_rate"]), 4), "ucl": round(float(v["ucl"]), 4),
                       "top_pattern": None if pd.isna(v.get("top_pattern")) else v.get("top_pattern")}
                      for k, v in hit.iterrows()]}


@app.post("/similar")
def similar(body: SimilarIn):
    wafer = parse(body.wafer_map)
    with torch.no_grad():
        vec = model().embed(torch.from_numpy(to_channels(wafer)[None])).numpy()[0]
    return {"k": body.k, "similar": similar_rows(vec, body.k)}


@app.get("/similar/{lot}/{wafer_index}")
def similar_known(lot: str, wafer_index: int, k: int = 10):
    meta = index().meta
    hit = np.flatnonzero((meta["lot"].to_numpy() == lot) & (meta["wafer_index"].to_numpy() == wafer_index))
    if not len(hit):
        raise HTTPException(404, f"no labeled wafer {lot}/{wafer_index}")
    row = meta.iloc[hit[0]]
    return {"query": {"lot": lot, "wafer_index": wafer_index, "label": row["label"]},
            "similar": similar_rows(index().vectors[hit[0]], k, exclude_lot=lot)}
