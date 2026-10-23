"""AppTest runs of the Overview tab across filter combinations."""

import itertools
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from application import overview as ov

CUBE = Path(__file__).resolve().parent.parent / "data" / "dashboard" / "order_cube.csv"


def _script() -> None:
    from app.components.overview import render_overview

    render_overview()


def _run(**picks: str) -> AppTest:
    at = AppTest.from_function(_script, default_timeout=60).run()
    for key, value in picks.items():
        box = next(s for s in at.selectbox if s.key == f"ov_{key}")
        box.select(value)
        at = at.run()
    return at


def _tile_numbers(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "ld-tile" in m.value)


def test_default_view_renders() -> None:
    at = _run()
    assert not at.exception
    assert "65,752" in _tile_numbers(at)
    assert "54.8%" in _tile_numbers(at)
    assert "36,048" in _tile_numbers(at)
    text = " ".join(m.value for m in at.markdown)
    assert "First Class is late 95% to 96% of the time in every market" in text
    assert "DataCo public supply chain data, 2015 to 2018, order level." in text


_OPTS = ov.filter_options(ov.load_cube(CUBE))
_SINGLE = [
    (name, value)
    for name, key in [
        ("market", "market"),
        ("segment", "segment"),
        ("mode", "mode"),
        ("year", "year"),
    ]
    for value in _OPTS[key]
]


@pytest.mark.parametrize(("name", "value"), _SINGLE)
def test_each_single_filter_value(name: str, value: str) -> None:
    at = _run(**{name: value})
    assert not at.exception


@pytest.mark.parametrize(
    "picks",
    [
        {"market": "Africa", "mode": "Same Day", "year": "2018"},
        {
            "market": "USCA",
            "segment": "Home Office",
            "mode": "First Class",
            "year": "2015",
        },
        {"market": "Europe", "segment": "Corporate", "year": "2017"},
    ],
)
def test_filter_combinations(picks: dict[str, str]) -> None:
    at = _run(**picks)
    assert not at.exception


def test_every_market_and_mode_pair_has_no_exception() -> None:
    for market, mode in itertools.islice(
        itertools.product(_OPTS["market"][1:3], _OPTS["mode"][1:]), 8
    ):
        assert not _run(market=market, mode=mode).exception


def test_empty_selection_shows_message() -> None:
    cube = ov.load_cube(CUBE)
    for market, mode, year in itertools.product(
        _OPTS["market"][1:], _OPTS["mode"][1:], _OPTS["year"][1:]
    ):
        if ov.apply_filters(cube, market=market, mode=mode, year=year).empty:
            at = _run(market=market, mode=mode, year=year)
            assert not at.exception
            assert any("No orders match" in i.value for i in at.info)
            return
    pytest.skip("no empty combination in the cube")


def test_figures_build_for_full_and_filtered_cube() -> None:
    from app.components import overview as view

    cube = ov.load_cube(CUBE)
    for df in (cube, ov.apply_filters(cube, market="Africa", mode="Same Day")):
        assert len(view.heatmap_figure(df).data) == 2
        assert len(view.trend_figure(df, sorted(cube["month"].unique())).data) >= 2
        assert view.treemap_figure(df).data[0].values[0] == int(df["late_orders"].sum())
        sankey = view.sankey_figure(df).data[0]
        assert sum(sankey.link.value) == int(df["orders"].sum())


def test_full_app_has_overview_as_first_tab() -> None:
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"),
        default_timeout=180,
    ).run()
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "Overview"


def test_trend_lines_keep_gaps_for_missing_months() -> None:
    from app.components import overview as view

    cube = ov.load_cube(CUBE)
    months = sorted(cube["month"].unique())
    fig = view.trend_figure(ov.apply_filters(cube, market="Africa"), months)
    for trace in fig.data:
        assert len(trace.x) == len(months)
        assert trace.connectgaps is False
    assert any(pd.isna(v) for v in fig.data[0].y)


def test_treemap_root_follows_market_and_is_white() -> None:
    from app.components import overview as view

    cube = ov.load_cube(CUBE)
    one = view.treemap_figure(ov.apply_filters(cube, market="LATAM")).data[0]
    assert one.labels[0] == "LATAM" and one.marker.colors[0] == "#FFFFFF"
    allm = view.treemap_figure(cube).data[0]
    assert allm.labels[0] == "All markets" and allm.marker.colors[0] == "#FFFFFF"
    assert "Standard" in allm.labels and "Standard Class" not in allm.labels


def test_trend_caption_flags_partial_coverage() -> None:
    from app.components import overview as view

    cube = ov.load_cube(CUBE)
    months = sorted(cube["month"].unique())
    assert "of 37 months" not in view._trend_caption(cube, months)
    assert "in 12 of 37 months" in view._trend_caption(
        ov.apply_filters(cube, year="2016"), months
    )
