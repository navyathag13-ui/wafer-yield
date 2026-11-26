import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from wafer_yield import api, cnn
from wafer_yield.features import features
from wafer_yield.maps import CLASS_INDEX, CLASSES
from wafer_yield.spc import lot_table, p_chart
from tests.synthetic import make


def test_cnn_shapes_and_probabilities():
    m = cnn.WaferCNN()
    probs = cnn.predict(m, [make("Center"), make("Scratch", n=30)])
    assert probs.shape == (2, len(CLASSES)) and np.allclose(probs.sum(1), 1, atol=1e-5)
    x = torch.zeros(4, 2, 64, 64)
    assert cnn.augment(x).shape == x.shape


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    torch.save(cnn.WaferCNN().state_dict(), tmp_path / "results" / "cnn.pt")
    rows = [{"lot": f"lot{i}", "wafer_index": 1, "dies": 1000, "fails": 400 if i == 3 else 100, "label": "none"}
            for i in range(1, 21)]
    p_chart(lot_table(pd.DataFrame(rows)), laney=False).to_parquet(tmp_path / "results" / "lot_spc.parquet")
    # A tiny tree model and similarity index, so every endpoint runs without real training.
    kinds = CLASSES * 3
    gbt = HistGradientBoostingClassifier(max_iter=5).fit(
        np.stack([features(make(k, seed=i)) for i, k in enumerate(kinds)]), [CLASS_INDEX[k] for k in kinds])
    joblib.dump(gbt, tmp_path / "results" / "gbt.joblib")
    (tmp_path / "results" / "ensemble.json").write_text(json.dumps({"cnn_weight": 0.5, "review_threshold": 0.99}))
    np.save(tmp_path / "results" / "embeddings.npy", np.random.default_rng(0).normal(size=(len(kinds), 64)))
    pd.DataFrame({"lot": [f"lot{i // 3}" for i in range(len(kinds))], "wafer_index": [i % 3 + 1 for i in range(len(kinds))],
                  "label": kinds, "split": "Training"}).to_parquet(tmp_path / "results" / "embedding_index.parquet")
    monkeypatch.setattr(api, "ROOT", tmp_path)
    clear()
    yield TestClient(api.app)
    clear()


def clear():
    for f in (api.model, api.chart, api.ensemble, api.index):
        f.cache_clear()


def test_classify(client):
    r = client.post("/classify", json={"wafer_map": make("Edge-Ring").tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["pattern"] in CLASSES and len(body["top3"]) == 3 and 0 < body["die_yield"] < 1
    assert body["needs_review"] is True          # threshold 0.99: an untrained model is never that sure
    assert abs(sum(t["p"] for t in body["top3"])) <= 1.0001
    assert body["guide"]["looks"] and all(c["url"].startswith("http") for c in body["guide"]["possible_causes"])


def test_classify_rejects_bad_maps(client):
    assert client.post("/classify", json={"wafer_map": [[0, 0], [0, 0]]}).status_code == 422
    assert client.post("/classify", json={"wafer_map": [[1, 5], [1, 1]]}).status_code == 422
    assert client.post("/classify", json={"wafer_map": [[1, 1], [1]]}).status_code == 422


def test_lot_and_signals(client):
    assert client.get("/lots/lot3").json()["out_of_control"] is True
    assert client.get("/lots/nope").status_code == 404
    s = client.get("/spc/signals").json()
    assert s["lots_checked"] == 20 and s["worst"][0]["lot"] == "lot3"


def test_similar_endpoints(client):
    r = client.post("/similar", json={"wafer_map": make("Center").tolist(), "k": 5}).json()
    assert len(r["similar"]) == 5 and r["similar"][0]["similarity"] >= r["similar"][-1]["similarity"]
    known = client.get("/similar/lot0/1?k=4").json()
    assert known["query"]["label"] == CLASSES[0]
    assert len(known["similar"]) == 4 and all(s["lot"] != "lot0" for s in known["similar"])
    assert client.get("/similar/lot0/9").status_code == 404
    assert client.post("/similar", json={"wafer_map": [[0]], "k": 3}).status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
