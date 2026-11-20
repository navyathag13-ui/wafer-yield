from wafer_yield.features import NAMES, features
from tests.synthetic import make


def f(kind):
    return dict(zip(NAMES, features(make(kind, n=48))))


def test_feature_vector_shape():
    assert features(make("none")).shape == (len(NAMES),)


def test_centre_and_edge_patterns_land_in_the_right_ring():
    c, e = f("Center"), f("Edge-Ring")
    assert c["ring_0.0_0.3"] > 0.8 and c["ring_0.92_1.01"] < 0.1
    assert e["ring_0.92_1.01"] > 0.8 and e["ring_0.0_0.3"] < 0.1


def test_scratch_is_long_and_loc_is_round():
    assert f("Scratch")["top_cluster_elongation"] > 5
    assert f("Loc")["top_cluster_elongation"] < 2


def test_random_has_many_clusters_and_even_sectors():
    r = f("Random")
    assert r["n_clusters"] > 20 and r["sector_spread"] < 0.05
