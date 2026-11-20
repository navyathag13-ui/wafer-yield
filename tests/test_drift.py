import numpy as np

from wafer_yield.defects import GUIDE, SOURCES
from wafer_yield.drift import DriftMonitor, nearest_distance, psi
from wafer_yield.maps import CLASSES


def probs_for(labels, conf, n_classes=9):
    p = np.full((len(labels), n_classes), (1 - conf) / (n_classes - 1))
    p[np.arange(len(labels)), labels] = conf
    return p


def test_psi_is_zero_for_same_mix_and_grows_with_shift():
    a = np.array([0.7, 0.2, 0.1])
    assert psi(a, a) < 1e-9
    assert psi(np.array([0.2, 0.7, 0.1]), a) > psi(np.array([0.6, 0.3, 0.1]), a) > 0


def test_nearest_distance():
    base = np.array([[1, 0], [0, 1]], dtype=np.float32)
    d = nearest_distance(np.array([[1, 0], [1, 1]]), base)
    assert d[0] < 1e-6 and abs(d[1] - (1 - np.sqrt(0.5))) < 1e-5


def test_monitor_quiet_on_normal_and_loud_on_shifted_batches():
    rng = np.random.default_rng(0)
    ref_labels = rng.choice(9, 4000, p=[.05, .02, .05, .1, .05, .01, .02, .02, .68])
    ref = probs_for(ref_labels, 0.9)
    ref_dist = rng.uniform(0.0, 0.05, 4000)
    mon = DriftMonitor(ref, ref_dist, review_threshold=0.8, batch=500)
    normal = mon.check(ref[rng.integers(0, 4000, 500)], rng.uniform(0.0, 0.05, 500))
    assert not normal["alarm"]
    shifted = mon.check(probs_for(rng.choice(9, 500), 0.6), rng.uniform(0.2, 0.4, 500))
    assert shifted["alarm"] and set(shifted["fired"]) == {"review_share", "mix_psi", "novelty"}


def test_defect_guide_covers_every_class_and_cites_known_sources():
    assert set(GUIDE) == set(CLASSES)
    for entry in GUIDE.values():
        assert entry["looks"]
        for _, src in entry["causes"]:
            assert src in SOURCES


def test_lot_batches_keep_lots_whole():
    from wafer_yield.drift import lot_batch
    lots = np.repeat(np.arange(50), 20)
    b = lot_batch(lots, 100, np.random.default_rng(0))
    assert len(b) == 100 and len(set(lots[b])) == 5   # five whole lots of 20
