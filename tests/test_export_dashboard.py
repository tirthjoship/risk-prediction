"""Tests for the dashboard export script and the committed order cube."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_dashboard import build_cube, build_order_table

ROOT = Path(__file__).resolve().parent.parent
CUBE = ROOT / "data" / "dashboard" / "order_cube.csv"
STATS = ROOT / "data" / "metrics" / "full_dataset_stats.json"


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order Id": [1, 1, 2, 3],
            "order date (DateOrders)": [
                "1/31/2018 22:56",
                "1/31/2018 22:56",
                "2/3/2018 01:00",
                "2/9/2018 10:30",
            ],
            "Shipping Mode": ["First Class"] * 3 + ["Standard Class"],
            "Market": ["Europe"] * 4,
            "Customer Segment": ["Consumer"] * 4,
            "Late_delivery_risk": [1, 1, 0, 1],
        }
    )


def test_line_items_collapse_to_one_order() -> None:
    orders = build_order_table(_raw())
    assert len(orders) == 3
    assert orders["late"].tolist() == [1, 0, 1]


def test_cube_counts_orders_not_lines() -> None:
    cube = build_cube(build_order_table(_raw()))
    assert int(cube["orders"].sum()) == 3
    jan = cube[cube["month"] == "2018-01"]
    assert int(jan["orders"].sum()) == 1
    assert int(jan["late_orders"].sum()) == 1
    assert set(cube.columns) == {
        "month",
        "year",
        "market",
        "shipping_mode",
        "customer_segment",
        "orders",
        "late_orders",
    }


@pytest.fixture(scope="module")
def cube() -> pd.DataFrame:
    return pd.read_csv(CUBE)


def test_cube_matches_known_totals(cube: pd.DataFrame) -> None:
    stats = json.loads(STATS.read_text())["dataset"]
    assert int(cube["orders"].sum()) == stats["total_orders"] == 65752
    assert int(cube["late_orders"].sum()) == stats["late_orders"]
    assert cube["late_orders"].sum() / cube["orders"].sum() == pytest.approx(
        0.548, abs=0.0005
    )


def test_cube_mode_rates_match_stats(cube: pd.DataFrame) -> None:
    stats = json.loads(STATS.read_text())["dataset"]["late_rate_by_mode"]
    by_mode = cube.groupby("shipping_mode")[["orders", "late_orders"]].sum()
    for mode, expected in stats.items():
        got = by_mode.loc[mode, "late_orders"] / by_mode.loc[mode, "orders"]
        assert got == pytest.approx(expected, abs=0.0005)


def test_cube_shape(cube: pd.DataFrame) -> None:
    assert cube["month"].nunique() == 37
    assert cube["month"].min() == "2015-01"
    assert cube["month"].max() == "2018-01"
    assert set(cube["market"]) == {"Europe", "Pacific Asia", "LATAM", "USCA", "Africa"}
    assert set(cube["customer_segment"]) == {"Consumer", "Corporate", "Home Office"}
    assert CUBE.stat().st_size < 500_000
    assert not cube.isna().any().any()
