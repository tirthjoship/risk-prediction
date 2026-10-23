"""Tests for the Overview tab data logic: filters, KPI math and titles."""

from pathlib import Path

import pandas as pd
import pytest

from application import overview as ov

CUBE = Path(__file__).resolve().parent.parent / "data" / "dashboard" / "order_cube.csv"


def _cube(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    """Rows are (month, market, mode, segment, orders, late_orders)."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "month",
            "market",
            "shipping_mode",
            "customer_segment",
            "orders",
            "late_orders",
        ],
    )
    frame["year"] = frame["month"].str[:4].astype(int)
    return frame


@pytest.fixture(scope="module")
def cube() -> pd.DataFrame:
    return ov.load_cube(CUBE)


def test_load_cube_rejects_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("month,orders\n2015-01,3\n")
    with pytest.raises(ValueError):
        ov.load_cube(bad)


def test_filter_options(cube: pd.DataFrame) -> None:
    opts = ov.filter_options(cube)
    assert opts["market"][0] == ov.ALL and len(opts["market"]) == 6
    assert len(opts["segment"]) == 4
    assert opts["mode"][:3] == [ov.ALL, "First Class", "Second Class"]
    assert opts["year"] == [ov.ALL, "2015", "2016", "2017", "2018"]


def test_filters_combine(cube: pd.DataFrame) -> None:
    assert ov.apply_filters(cube)["orders"].sum() == 65752
    sub = ov.apply_filters(cube, market="Europe", mode="First Class", year="2016")
    assert set(sub["market"]) == {"Europe"}
    assert set(sub["shipping_mode"]) == {"First Class"}
    assert set(sub["year"]) == {2016}
    parts = sum(
        ov.apply_filters(cube, segment=s)["orders"].sum()
        for s in ov.filter_options(cube)["segment"][1:]
    )
    assert parts == 65752


def test_kpis_full_cube(cube: pd.DataFrame) -> None:
    k = ov.kpis(cube)
    assert k["orders"] == 65752
    assert k["late_rate"] == pytest.approx(0.548, abs=0.0005)
    assert k["late_orders"] == 36048
    assert 0 < k["fast_share"] < 1


def test_kpis_small_and_empty() -> None:
    df = _cube(
        [
            ("2015-01", "Europe", "First Class", "Consumer", 10, 9),
            ("2015-01", "Europe", "Standard Class", "Consumer", 30, 10),
        ]
    )
    k = ov.kpis(df)
    assert k == {
        "orders": 40,
        "late_orders": 19,
        "late_rate": 19 / 40,
        "fast_share": 0.25,
    }
    empty = ov.kpis(df.iloc[0:0])
    assert empty["late_rate"] == 0.0 and empty["orders"] == 0


def test_helpers() -> None:
    assert ov.rate(1, 0) == 0.0
    assert ov.pct(0.548) == "55%"
    assert ov.span_text(37) == "three years"
    assert ov.span_text(12) == "one year"
    assert ov.span_text(7) == "7 months"
    assert ov.describe({"market": "Europe", "year": ov.ALL}) == "market: Europe"
    assert ov.describe({"market": ov.ALL}) == "all orders"


def test_heatmap_title_same_everywhere() -> None:
    rows = []
    for market, late in [("Europe", 95), ("LATAM", 96)]:
        rows.append(("2015-01", market, "First Class", "Consumer", 100, late))
        rows.append(("2015-01", market, "Standard Class", "Consumer", 100, 38))
    assert ov.heatmap_title(_cube(rows)) == (
        "First Class is late 95% to 96% of the time in every market"
    )


def test_heatmap_title_collapses_equal_rounding() -> None:
    rows = [
        ("2015-01", "Europe", "First Class", "Consumer", 1000, 951),
        ("2015-01", "LATAM", "First Class", "Consumer", 1000, 953),
        ("2015-01", "Europe", "Standard Class", "Consumer", 100, 38),
    ]
    assert (
        ov.heatmap_title(_cube(rows))
        == "First Class is late 95% of the time in every market"
    )


def test_heatmap_title_neutral_when_spread_is_wide() -> None:
    rows = [
        ("2015-01", "Europe", "First Class", "Consumer", 100, 99),
        ("2015-01", "LATAM", "First Class", "Consumer", 100, 80),
        ("2015-01", "Europe", "Standard Class", "Consumer", 100, 38),
    ]
    title = ov.heatmap_title(_cube(rows))
    assert "every market" not in title
    assert title == "Late rate runs from 38% to 99% by market and mode"


def test_heatmap_title_single_market_and_empty(cube: pd.DataFrame) -> None:
    title = ov.heatmap_title(ov.apply_filters(cube, market="Europe"))
    assert title == "First Class is late 96% of the time in Europe"
    assert ov.heatmap_title(cube.iloc[0:0]) == "No orders match these filters"


def test_heatmap_title_ignores_thin_cells() -> None:
    rows = [
        ("2015-01", "Europe", "First Class", "Consumer", 1000, 950),
        ("2015-01", "LATAM", "First Class", "Consumer", 10, 2),
        ("2015-01", "USCA", "First Class", "Consumer", 1000, 960),
    ]
    assert "every market" in ov.heatmap_title(_cube(rows))


def _months(n: int) -> list[str]:
    return [f"{2015 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def test_trend_title_steady() -> None:
    rows = []
    for i, m in enumerate(_months(36)):
        rows.append((m, "Europe", "First Class", "Consumer", 100, 95 + (i % 2)))
        rows.append((m, "Europe", "Standard Class", "Consumer", 100, 38 + (i % 3)))
    assert (
        ov.trend_title(_cube(rows)) == "Each shipping mode holds steady for three years"
    )


def test_trend_title_not_steady_beyond_threshold() -> None:
    rows = []
    for i, m in enumerate(_months(36)):
        rows.append((m, "Europe", "First Class", "Consumer", 100, 95))
        rows.append((m, "Europe", "Same Day", "Consumer", 100, 30 + (i % 2) * 30))
    title = ov.trend_title(_cube(rows))
    assert "steady for" not in title
    assert (
        title
        == "First Class stays steady while Same Day swings from 30% to 60% month to month"
    )


def test_trend_title_skips_thin_months() -> None:
    rows = []
    for m in _months(24):
        rows.append((m, "Europe", "First Class", "Consumer", 100, 95))
    # A 10-order month with a wild rate must not break the steady claim.
    rows.append(("2017-01", "Europe", "First Class", "Consumer", 10, 1))
    assert ov.trend_title(_cube(rows)) == "First Class holds steady for two years"


def test_trend_title_edge_cases() -> None:
    thin = _cube([("2015-01", "Europe", "First Class", "Consumer", 10, 5)])
    assert ov.trend_title(thin) == "Not enough orders per month to show a trend"
    rows = [
        (m, "Europe", "Same Day", "Consumer", 100, 30 + (i % 2) * 30)
        for i, m in enumerate(_months(6))
    ]
    assert (
        ov.trend_title(_cube(rows))
        == "Same Day moves between 30% and 60% month to month"
    )


def test_treemap_and_sankey_titles(cube: pd.DataFrame) -> None:
    assert (
        ov.treemap_title(cube)
        == "Biggest pool of late orders: Europe Standard Class (4,216)"
    )
    fast_late = cube[cube["shipping_mode"].isin(ov.FAST_MODES)]["late_orders"].sum()
    expected = f"{fast_late / cube['late_orders'].sum() * 100:.0f}% of late orders shipped First or Second Class"
    assert ov.sankey_title(cube) == expected
    none = cube.copy()
    none["late_orders"] = 0
    assert ov.treemap_title(none) == "No late orders in this selection"
    assert ov.sankey_title(none) == "No late orders in this selection"


def test_tables(cube: pd.DataFrame) -> None:
    assert ov.market_order(cube)[0] == "Europe"
    assert ov.mode_order(cube) == ov.MODE_ORDER
    flow = ov.mode_flow(cube)
    assert (flow["on_time"] + flow["late_orders"] == flow["orders"]).all()
    assert flow["orders"].sum() == 65752
    assert ov.monthly_overall(cube)["month"].nunique() == 37
    assert len(ov.market_mode_table(cube)) == 20
