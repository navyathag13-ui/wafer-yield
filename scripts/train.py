"""Train and score the two pattern classifiers on WM-811K's own Training / Test split.

    python scripts/train.py [--epochs 15]

Baseline: gradient-boosted trees on the hand-made features (wafer_yield/features.py).
CNN:      wafer_yield/cnn.py, with 10% of Training held back to pick the best epoch.
Ensemble: weighted average of the two, weight and "needs review" threshold picked on validation.
Similar-wafer search: CNN embeddings, scored by precision@10 against plain pixel similarity.
Everything is scored once on the Test split. Writes results/metrics.json, cnn.pt, gbt.joblib,
ensemble.json, embeddings.npy, embedding_index.parquet and confusion_ensemble.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wafer_yield import cnn  # noqa: E402
from wafer_yield.etl import decode  # noqa: E402
from wafer_yield.features import features  # noqa: E402
from wafer_yield.maps import CLASS_INDEX, CLASSES, FAIL, OFF, pool  # noqa: E402
from wafer_yield.similar import precision_at_k  # noqa: E402


def load():
    df = pd.read_parquet(ROOT / "data" / "wafers.parquet",
                         filters=[("label", "!=", "__none__")])  # pyarrow drops null labels here
    df = df[df["label"].notna() & df["split"].isin(["Training", "Test"])].reset_index(drop=True)
    maps = [decode(r) for _, r in df[["map", "rows", "cols"]].iterrows()]
    y = df["label"].map(CLASS_INDEX).to_numpy()
    return df, maps, y


def report(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    return {"accuracy": round(rep["accuracy"], 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "macro_f1_defects_only": round(f1_score(y_true, y_pred, labels=range(8), average="macro"), 4),
            "per_class": {c: {k: round(v, 4) for k, v in rep[c].items()} for c in CLASSES}}


def train_cnn(x_tr, y_tr, x_val, y_val, epochs: int, dev):
    model = cnn.WaferCNN().to(dev)
    counts = np.bincount(y_tr, minlength=len(CLASSES))
    # Inverse square-root weights: 'none' is ~85% of labels, but full inverse weighting
    # over-rewards guessing the rare classes.
    w = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(y_tr) // 256 + 1))
    best, best_state, history = -1.0, None, []
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb = cnn.augment(channels(xt[idx]).to(dev))
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx].to(dev))
            loss.backward()
            opt.step()
            sched.step()
        pred = predict_codes(model, x_val, dev).argmax(1)
        f1 = f1_score(y_val, pred, average="macro")
        history.append({"epoch": ep + 1, "loss": round(float(loss.detach()), 4), "val_macro_f1": round(f1, 4)})
        print(history[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def channels(codes: torch.Tensor) -> torch.Tensor:
    """uint8 code maps (N, 64, 64) -> float channels (N, 2, 64, 64). Kept as codes to save memory."""
    return torch.stack([(codes > OFF), (codes == FAIL)], dim=1).float()


@torch.no_grad()
def predict_codes(model, codes: np.ndarray, dev) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(codes), 1024):
        out.append(torch.softmax(model(channels(torch.from_numpy(codes[i:i + 1024])).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def main(epochs: int) -> None:
    t0 = time.perf_counter()
    df, maps, y = load()
    tr, te = (df["split"] == "Training").to_numpy(), (df["split"] == "Test").to_numpy()
    print(f"labeled wafers: {len(df)} (Training {tr.sum()}, Test {te.sum()}); load {time.perf_counter() - t0:.0f}s")

    # Hold out whole lots, like the official Training/Test split does. Wafers in one lot share
    # their defect, so a random wafer split leaks: it scored 0.937 here against 0.675 on Test.
    # Both models train on `fit` only, so validation can fairly pick the ensemble weight and
    # the review threshold. Test is touched once, at the end.
    rng = np.random.default_rng(0)
    tr_idx = np.flatnonzero(tr)
    lots = df["lot"].to_numpy()
    tr_lots = np.unique(lots[tr_idx])
    rng.shuffle(tr_lots)
    val_lots = set(tr_lots[: len(tr_lots) // 10])
    is_val = np.array([lots[i] in val_lots for i in tr_idx])
    val_idx, fit_idx = tr_idx[is_val], tr_idx[~is_val]
    te_idx = np.flatnonzero(te)

    t1 = time.perf_counter()
    feats = np.stack([features(m) for m in maps])
    gbt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0)
    gbt.fit(feats[fit_idx], y[fit_idx])
    p_gbt_val, p_gbt_te = gbt.predict_proba(feats[val_idx]), gbt.predict_proba(feats[te_idx])
    base = report(y[te_idx], p_gbt_te.argmax(1))
    base["seconds"] = round(time.perf_counter() - t1, 1)
    print("baseline macro F1", base["macro_f1"])

    # pool(), not resize(): shrinking a big map must not drop a one-die-wide scratch.
    codes = np.stack([pool(m) for m in maps]).astype(np.uint8)
    torch.manual_seed(0)
    dev = cnn.device()
    t2 = time.perf_counter()
    model, history = train_cnn(codes[fit_idx], y[fit_idx], codes[val_idx], y[val_idx], epochs, dev)
    p_cnn_val, p_cnn_te = predict_codes(model, codes[val_idx], dev), predict_codes(model, codes[te_idx], dev)
    deep = report(y[te_idx], p_cnn_te.argmax(1))
    deep["seconds"] = round(time.perf_counter() - t2, 1)
    deep["history"] = history
    deep["device"] = str(dev)
    print("cnn macro F1", deep["macro_f1"])

    # Ensemble: weighted average of the two models' probabilities, weight picked on validation.
    weights = np.round(np.linspace(0, 1, 11), 2)
    val_f1 = [f1_score(y[val_idx], (w * p_cnn_val + (1 - w) * p_gbt_val).argmax(1), average="macro") for w in weights]
    w_best = float(weights[int(np.argmax(val_f1))])
    p_ens_val = w_best * p_cnn_val + (1 - w_best) * p_gbt_val
    p_ens_te = w_best * p_cnn_te + (1 - w_best) * p_gbt_te
    ens = report(y[te_idx], p_ens_te.argmax(1))
    ens["cnn_weight"] = w_best
    ens["val_macro_f1_by_weight"] = dict(zip(map(str, weights), np.round(val_f1, 4).tolist()))
    print("ensemble weight", w_best, "macro F1", ens["macro_f1"])

    # "Not sure" flag: wafers whose top probability is below t go to an engineer. t is the
    # highest threshold that sends at most 5% of validation wafers to review.
    conf_val = p_ens_val.max(1)
    t_review = float(np.quantile(conf_val, 0.05))
    review = review_table(y[te_idx], p_ens_te, [0.5, 0.6, 0.7, 0.8, 0.9, round(t_review, 3)])
    print("review threshold", round(t_review, 3), review[-1])

    t3 = time.perf_counter()
    emb = embed_codes(model, codes, dev)
    sim = similarity_eval(emb, codes, y, fit_idx, te_idx)
    sim["seconds"] = round(time.perf_counter() - t3, 1)
    print("similar wafers precision@10", sim["embedding"]["mean_defects"], "vs pixels", sim["pixels"]["mean_defects"])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "cnn.pt")
    joblib.dump(gbt, out / "gbt.joblib")
    (out / "ensemble.json").write_text(json.dumps({"cnn_weight": w_best, "review_threshold": round(t_review, 4)}))
    np.save(out / "embeddings.npy", emb.astype(np.float16))
    df[["lot", "wafer_index", "label", "split"]].to_parquet(out / "embedding_index.parquet")
    cm = confusion_matrix(y[te_idx], p_ens_te.argmax(1), labels=range(len(CLASSES)))
    metrics = {"train": int(len(fit_idx)), "val": int(len(val_idx)), "test": int(len(te_idx)),
               "test_label_counts": {c: int((y[te_idx] == i).sum()) for i, c in enumerate(CLASSES)},
               "baseline_gbt": base, "cnn": deep, "ensemble": ens, "review": review,
               "review_threshold": round(t_review, 4), "similar_wafers": sim,
               "ensemble_confusion": cm.tolist(), "cnn_confusion":
                   confusion_matrix(y[te_idx], p_cnn_te.argmax(1), labels=range(len(CLASSES))).tolist(),
               "params": int(sum(p.numel() for p in model.parameters()))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, out / "confusion_ensemble.png")


def review_table(y_true, probs, thresholds) -> list[dict]:
    """For each threshold: share of wafers sent to review, and how the rest score."""
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({"threshold": t, "review_share": round(float(1 - auto.mean()), 4),
                     "auto_accuracy": round(float((pred[auto] == y_true[auto]).mean()), 4),
                     "auto_macro_f1": round(f1_score(y_true[auto], pred[auto], average="macro"), 4),
                     "defects_reviewed_share": round(float((~auto & (y_true != CLASS_INDEX["none"])).sum()
                                                           / max((y_true != CLASS_INDEX["none"]).sum(), 1)), 4)})
    return rows


@torch.no_grad()
def embed_codes(model, codes: np.ndarray, dev) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(codes), 1024):
        out.append(model.embed(channels(torch.from_numpy(codes[i:i + 1024])).to(dev)).cpu().numpy())
    return np.concatenate(out)


def similarity_eval(emb, codes, y, base_idx, query_idx, k: int = 10) -> dict:
    """Test-split defect wafers search the training wafers (different lots by construction).
    Score: share of the 10 nearest that carry the same label. Compared with plain pixel
    similarity on 32 x 32 fail masks, to show what the CNN embedding adds."""
    q = query_idx[y[query_idx] != CLASS_INDEX["none"]]
    pix = codes[:, ::2, ::2].reshape(len(codes), -1) == FAIL
    res = {}
    for name, vecs in (("embedding", emb), ("pixels", pix.astype(np.float32))):
        p = precision_at_k(vecs[q], y[q], vecs[base_idx], y[base_idx], k)
        per = {CLASSES[c]: round(float(p[y[q] == c].mean()), 4) for c in range(8) if (y[q] == c).any()}
        res[name] = {"mean_defects": round(float(p.mean()), 4), "per_class": per}
    res["queries"] = int(len(q))
    res["k"] = k
    return res


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalised recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    main(ap.parse_args().epochs)
