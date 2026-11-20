import numpy as np
import pandas as pd

from wafer_yield.etl import check, decode, process, unwrap
from wafer_yield.maps import CLASSES, resize, to_channels
from tests.synthetic import make


def rec(wafer, lot="lot1", idx=1.0, label="Center", split="Training", die_size=None):
    w = np.asarray(wafer)
    return {"waferMap": w, "dieSize": float((w > 0).sum()) if die_size is None else die_size, "lotName": lot,
            "waferIndex": idx, "trianTestLabel": np.array([[split]]) if split else np.zeros((0, 0)),
            "failureType": np.array([[label]]) if label else np.zeros((0, 0))}


def test_unwrap_handles_nested_and_empty():
    assert unwrap(np.array([["Edge-Ring"]])) == "Edge-Ring"
    assert unwrap(np.zeros((0, 0))) is None
    assert unwrap([]) is None


def test_check_rejects_bad_maps():
    assert check(np.zeros((5, 5)))[1] == "no dies on the map"
    assert "unknown die codes" in check(np.array([[1, 3], [1, 1]]))[1]
    assert "not a 2D" in check(np.array([1, 2, 1]))[1]
    ok, why = check(make("Center"))
    assert why is None and ok.dtype == np.uint8


def test_process_counts_add_up_and_round_trip(tmp_path):
    good = make("Center")
    recs = [rec(good), rec(make("none"), idx=2.0, label=None, split=None),
            rec(np.zeros((4, 4)), idx=3.0), rec(good, label="Banana", idx=4.0),
            rec(good, idx=1.0),                      # duplicate of the first wafer
            rec(good, idx=30.0, die_size=5.0)]       # index out of range and wrong dieSize
    stats = process(recs, tmp_path)
    assert stats["input"] == 6 and stats["loaded"] == 4 and stats["quarantined"] == 2
    assert stats["labeled"] == 3 and stats["unlabeled"] == 1
    assert stats["flag:duplicate_lot_index"] == 1
    assert stats["flag:die_count_mismatch"] == 1 and stats["flag:wafer_index_out_of_range"] == 1
    df = pd.read_parquet(tmp_path / "wafers.parquet")
    assert np.array_equal(decode(df.iloc[0]), good)
    assert abs(df.iloc[0]["yield"] - (good == 1).sum() / (good > 0).sum()) < 1e-6
    q = pd.read_parquet(tmp_path / "quarantine.parquet")
    assert sorted(q["source_row"]) == [2, 3]


def test_resize_keeps_codes():
    w = make("Random", n=37)
    r = resize(w)
    assert r.shape == (64, 64) and set(np.unique(r)) <= {0, 1, 2}
    ch = to_channels(w)
    assert ch.shape == (2, 64, 64) and ch[1].sum() <= ch[0].sum()


def test_all_synthetic_kinds_are_known_classes():
    for k in CLASSES:
        assert check(make(k))[1] is None


def test_pool_keeps_a_thin_scratch_that_nearest_drops():
    from wafer_yield.maps import pool
    w = np.ones((150, 150), dtype=np.uint8)
    w[:, 76] = 2                                   # one-die-wide scratch on a column nearest skips
    assert (resize(w) == 2).sum() == 0             # nearest samples columns 75 and 77, not 76
    assert (pool(w) == 2).sum() == 64              # pooled keeps one full column
    small = make("Center", n=30)
    assert np.array_equal(pool(small), resize(small))  # enlarging: same as nearest
