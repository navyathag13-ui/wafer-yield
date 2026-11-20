import numpy as np
import pandas as pd

from wafer_yield.spc import lot_table, p_chart, run_rule


def wafers(fail_rates, dies=1000, per_lot=5):
    rows = []
    for i, p in enumerate(fail_rates, start=1):
        for w in range(per_lot):
            rows.append({"lot": f"lot{i}", "wafer_index": w + 1, "dies": dies, "fails": int(p * dies),
                         "label": "Center" if p > 0.3 else "none"})
    return pd.DataFrame(rows)


def test_lot_table_sorts_by_lot_number_and_sums():
    lots = lot_table(wafers([0.1, 0.2, 0.1] + [0.1] * 8))
    assert list(lots["lot"][:3]) == ["lot1", "lot2", "lot3"] and lots["lot"].iloc[-1] == "lot11"
    assert lots.loc[1, "dies"] == 5000 and abs(lots.loc[1, "fail_rate"] - 0.2) < 1e-9


def test_p_chart_flags_the_bad_lot():
    rates = [0.10] * 30
    rates[12] = 0.45
    chart = p_chart(lot_table(wafers(rates)), laney=False)
    assert chart["out_of_control"].sum() == 1 and chart.loc[12, "out_of_control"]
    assert chart.loc[12, "top_pattern"] == "Center"


def test_laney_widens_limits_when_lots_vary_more_than_binomial():
    rng = np.random.default_rng(1)
    rates = np.clip(rng.normal(0.1, 0.03, 200), 0.01, 0.5)  # real-fab-like lot to lot spread
    plain = p_chart(lot_table(wafers(rates, dies=5000)), laney=False)
    laney = p_chart(lot_table(wafers(rates, dies=5000)), laney=True)
    assert laney["sigma_z"].iloc[0] > 3
    assert plain["out_of_control"].mean() > 0.3 > laney["out_of_control"].mean()


def test_run_rule_needs_eight_in_a_row():
    chart = pd.DataFrame({"fail_rate": [0.2] * 7 + [0.0] + [0.2] * 8, "center": 0.1})
    hits = run_rule(chart)
    assert not hits[:8].any() and hits.iloc[-1] and hits.sum() == 1


def test_per_product_charts_remove_product_mix():
    # Two products, each perfectly stable, at different normal fail rates (5% and 20%).
    rows = []
    for i in range(1, 61):
        rate, shape = (0.05, (30, 30)) if i % 2 else (0.20, (40, 40))
        rows.append({"lot": f"lot{i}", "wafer_index": 1, "dies": 2000, "fails": int(rate * 2000),
                     "rows": shape[0], "cols": shape[1], "label": "none"})
    lots = lot_table(pd.DataFrame(rows))
    assert set(lots["product"]) == {"30x30", "40x40"}
    one_chart = p_chart(lots, laney=False, by=None)
    per_product = p_chart(lots, laney=False)
    assert one_chart["out_of_control"].all()          # every lot looks wrong on a mixed chart
    assert not per_product["out_of_control"].any()    # none do once products are separated
    assert not run_rule(per_product).any() and run_rule(one_chart, by=None).sum() == 0  # alternating sides


def test_small_products_are_not_charted():
    rows = [{"lot": f"lot{i}", "wafer_index": 1, "dies": 1000, "fails": 100, "rows": 10, "cols": 10}
            for i in range(5)]
    chart = p_chart(lot_table(pd.DataFrame(rows)))
    assert not chart["charted"].any() and not chart["out_of_control"].any()
