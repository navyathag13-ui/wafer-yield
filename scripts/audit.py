"""Label audit and drift check, run after scripts/train.py.

    python scripts/audit.py

Label audit: WM-811K's labels were made by people, so some are wrong. A wafer is a suspect
when at least 8 of its 10 most similar labeled wafers from OTHER lots share one different
label, and the ensemble also gives that label at least 0.8. A suspect is not proven wrong:
it is a wafer worth a second look. Writes results/label_audit.parquet, label_audit.png and
the "label_audit" part of results/audit.json.

Drift check: measures false alarms of wafer_yield/drift.py with 5-fold cross-validation over
validation lots, then calibrates on all validation lots and streams the Test split in lot order. Writes the "drift" part
of results/audit.json and results/drift.png.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wafer_yield.drift import METRICS, DriftMonitor, lot_batch, nearest_distance  # noqa: E402
from wafer_yield.etl import decode  # noqa: E402
from wafer_yield.maps import CLASS_INDEX, CLASSES  # noqa: E402
from wafer_yield.similar import normalize  # noqa: E402

OUT = ROOT / "results"
CMAP = ListedColormap(["#ffffff", "#9ecae1", "#d62728"])
K, VOTES, MIN_P = 10, 8, 0.8


def neighbours(emb: np.ndarray, lots: np.ndarray, k: int = K, batch: int = 512) -> np.ndarray:
    """Indices of each row's k most similar rows from other lots."""
    e = normalize(emb)
    out = np.empty((len(e), k), dtype=np.int64)
    for i in range(0, len(e), batch):
        sims = e[i:i + batch] @ e.T
        cand = np.argpartition(-sims, 60, axis=1)[:, :60]
        for r, row in enumerate(cand):
            row = row[np.argsort(-sims[r, row])]
            row = row[lots[row] != lots[i + r]][:k]
            out[i + r, :len(row)] = row
    return out


def label_audit(emb, pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = pred["label"].to_numpy()
    nb = neighbours(emb, pred["lot"].to_numpy())
    nb_labels = labels[nb]
    vote = [Counter(r).most_common(1)[0] for r in nb_labels]
    pred["vote_label"] = [v[0] for v in vote]
    pred["vote_count"] = [v[1] for v in vote]
    probs = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    pred["p_vote"] = probs[np.arange(len(pred)), pred["vote_label"].map(CLASS_INDEX).to_numpy()]
    sus = (pred["vote_label"] != pred["label"]) & (pred["vote_count"] >= VOTES) & (pred["p_vote"] >= MIN_P)
    pred["suspect"] = sus
    s = pred[sus]
    test = pred[pred["role"] == "test"]
    y, yhat = test["label"].map(CLASS_INDEX), probs[test.index].argmax(1)
    keep = ~test["suspect"].to_numpy()
    summary = {
        "rule": f">= {VOTES} of {K} nearest other-lot wafers agree on a different label and the ensemble gives it >= {MIN_P}",
        "labeled_wafers": int(len(pred)), "suspects": int(sus.sum()),
        "suspect_share": round(float(sus.mean()), 4),
        "by_role": s["role"].value_counts().to_dict(),
        "given_to_suggested": {f"{a} -> {b}": int(n) for (a, b), n in
                               s.groupby(["label", "vote_label"]).size().sort_values(ascending=False).head(12).items()},
        "test_macro_f1_all": round(f1_score(y, yhat, average="macro"), 4),
        "test_macro_f1_without_suspects": round(f1_score(y[keep], yhat[keep], average="macro"), 4),
        "note": "Dropping suspects favours the model (it agrees with their suggested label), so the second F1 is an upper bound, not a corrected score.",
    }
    return summary, s


def plot_suspects(s: pd.DataFrame, path: Path, n: int = 24, seed: int = 0) -> pd.DataFrame:
    wafers = pd.read_parquet(ROOT / "data" / "wafers.parquet", columns=["lot", "wafer_index", "rows", "cols", "map"])
    sample = s.sample(min(n, len(s)), random_state=seed)
    m = sample.merge(wafers, on=["lot", "wafer_index"], how="left")
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(m) / cols)), cols, figsize=(16, 2.3 * np.ceil(len(m) / cols)))
    for ax, (_, r) in zip(np.ravel(axes), m.iterrows()):
        ax.imshow(decode(r), cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"labeled {r['label']}\nlooks {r['vote_label']} ({r['vote_count']}/10)", fontsize=7)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.suptitle("Random label-audit suspects: the given label, and what 10 similar wafers from other lots say", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return sample


def drift_check(emb, pred: pd.DataFrame, review_threshold: float, batch: int = 500, quantile: float = 0.999) -> dict:
    """False alarms come from 5-fold cross-validation over validation lots (calibrate on four
    folds, draw normal batches from the fifth). The final monitor is calibrated on all
    validation lots and then reads the Test split as a stream, in lot order."""
    probs = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    roles, lots = pred["role"].to_numpy(), pred["lot"].to_numpy()
    dist = nearest_distance(emb, emb[roles == "fit"])
    val = np.flatnonzero(roles == "val")
    rng = np.random.default_rng(1)
    val_lots = np.unique(lots[val])
    rng.shuffle(val_lots)
    false = []
    for fold in np.array_split(val_lots, 5):
        held = val[np.isin(lots[val], fold)]
        cal = val[~np.isin(lots[val], fold)]
        m = DriftMonitor(probs[cal], dist[cal], review_threshold, ref_lots=lots[cal], batch=batch, quantile=quantile)
        false += [m.check(probs[held][i], dist[held][i])["alarm"] for i in (lot_batch(lots[held], batch, rng) for _ in range(100))]
    mon = DriftMonitor(probs[val], dist[val], review_threshold, ref_lots=lots[val], batch=batch, quantile=quantile)
    test = np.flatnonzero(roles == "test")
    order = test[np.argsort(pd.to_numeric(pd.Series(lots[test]).str.extract(r"(\d+)")[0]).to_numpy(), kind="stable")]
    stream = [mon.check(probs[order[i:i + batch]], dist[order[i:i + batch]])
              for i in range(0, len(order) - batch + 1, batch)]
    fired = Counter(f for s in stream for f in s["fired"])
    plot_drift(stream, mon.thresholds, OUT / "drift.png")
    return {"batch": batch, "quantile": quantile, "thresholds": {k: round(v, 4) for k, v in mon.thresholds.items()},
            "false_alarm_rate_5fold_validation_lots": round(float(np.mean(false)), 4),
            "test_batches": len(stream), "test_batches_alarmed": int(sum(s["alarm"] for s in stream)),
            "test_alarm_rate": round(float(np.mean([s["alarm"] for s in stream])), 4),
            "fired_by_metric": dict(fired),
            "median_test_batch": {k: round(float(np.median([s[k] for s in stream])), 4) for k in METRICS},
            "median_validation_wafer_novelty": round(float(np.median(dist[val])), 4),
            "validation_review_share": round(float((probs[val].max(1) < review_threshold).mean()), 4)}


def plot_drift(stream, thresholds, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    for ax, m in zip(axes, METRICS):
        ax.plot([s[m] for s in stream], ".-", ms=3, lw=0.8, color="#333")
        ax.axhline(thresholds[m], color="#d62728", ls="--", lw=1, label="alarm line (99.9th pct of normal)")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("Test batch of 500 wafers, in lot order", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    emb = np.load(OUT / "embeddings.npy").astype(np.float32)
    pred = pd.read_parquet(OUT / "labeled_predictions.parquet")
    cfg = json.loads((OUT / "ensemble.json").read_text())
    audit, suspects = label_audit(emb, pred.copy())
    suspects.to_parquet(OUT / "label_audit.parquet")
    sample = plot_suspects(suspects, OUT / "label_audit.png")
    sample[["lot", "wafer_index", "label", "vote_label", "vote_count", "p_vote", "role"]].to_csv(
        OUT / "label_audit_sample.csv", index=False)
    drift = drift_check(emb, pred, cfg["review_threshold"])
    (OUT / "audit.json").write_text(json.dumps({"label_audit": audit, "drift": drift}, indent=2))
    print(json.dumps({"label_audit": audit, "drift": drift}, indent=2))


if __name__ == "__main__":
    main()
