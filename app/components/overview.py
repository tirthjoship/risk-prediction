"""Overview tab: filters, KPI band and four linked charts in a 2x2 board.

All numbers come from the order cube at runtime. Titles are computed by
application.overview so they always match the filtered data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import ui_theme
from application import overview as ov

CUBE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dashboard"
CUBE_FILE = CUBE_PATH / "order_cube.csv"

CHART_HEIGHT = 310
MARGIN = {"l": 4, "r": 4, "t": 6, "b": 4}
# Sequential scale for late rate: pale cream, amber, brand red.
RATE_SCALE = [[0.0, "#F7F1E3"], [0.5, "#D9A441"], [1.0, "#B3261E"]]
MODE_COLORS = {
    "First Class": ui_theme.RED,
    "Second Class": ui_theme.AMBER,
    "Same Day": "#7A8794",
    "Standard Class": ui_theme.NAVY,
}
HEATMAP_KEY = "ov_heatmap"
FILTER_KEYS = {
    "market": "ov_market",
    "segment": "ov_segment",
    "mode": "ov_mode",
    "year": "ov_year",
}


@st.cache_data
def load_order_cube() -> pd.DataFrame:
    """Load the committed order cube once per session."""
    return ov.load_cube(CUBE_FILE)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _colorbar() -> dict[str, object]:
    return {
        "thickness": 10,
        "len": 0.75,
        "tickformat": ".0%",
        "tickfont": {"size": 11},
        "outlinewidth": 0,
    }


# ---------------------------------------------------------------------------
# Figure builders (pure: DataFrame in, Figure out)
# ---------------------------------------------------------------------------
def heatmap_figure(df: pd.DataFrame) -> go.Figure:
    """Market by shipping mode matrix, colored and labeled by late rate."""
    table = ov.market_mode_table(df)
    markets = ov.market_order(df)
    modes = ov.mode_order(df)
    rates = table.pivot(index="market", columns="shipping_mode", values="late_rate")
    orders = table.pivot(index="market", columns="shipping_mode", values="orders")
    rates = rates.reindex(index=markets, columns=modes)
    orders = orders.reindex(index=markets, columns=modes)
    lo, hi = float(table["late_rate"].min()), float(table["late_rate"].max())
    if hi - lo < 0.02:
        lo, hi = max(0.0, lo - 0.01), min(1.0, hi + 0.01)
    labels = [m.replace(" ", "<br>") for m in modes]
    hover = [
        [
            (
                ""
                if pd.isna(rates.loc[mk, md])
                else f"{mk}, {md}<br>Late rate {rates.loc[mk, md]:.1%}"
                f"<br>{int(orders.loc[mk, md]):,} orders"
            )
            for md in modes
        ]
        for mk in markets
    ]
    fig = go.Figure(
        go.Heatmap(
            z=rates.values,
            x=labels,
            y=markets,
            zmin=lo,
            zmax=hi,
            colorscale=RATE_SCALE,
            xgap=3,
            ygap=3,
            texttemplate="%{z:.0%}",
            textfont={"size": 13},
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar={
                **_colorbar(),
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.02,
                "yanchor": "top",
                "len": 0.6,
            },
        )
    )
    # Heatmap cells cannot be selected, so an invisible marker sits on each
    # cell. Streamlit reads the clicked marker's market from it.
    fig.add_trace(
        go.Scatter(
            x=[lb for _ in markets for lb in labels],
            y=[mk for mk in markets for _ in labels],
            mode="markers",
            marker={
                "size": min(110, int(250 / max(len(markets), 1))),
                "opacity": 0.0,
                "symbol": "square",
            },
            selected={"marker": {"opacity": 0.0}},
            unselected={"marker": {"opacity": 0.0}},
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.update_layout(clickmode="event+select", dragmode=False)
    fig.update_layout(
        **ui_theme.plotly_layout(
            height=CHART_HEIGHT, margin={"l": 4, "r": 4, "t": 6, "b": 36}
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    fig.update_xaxes(showgrid=False, side="top", tickangle=0, tickfont={"size": 11})
    return fig


def trend_figure(df: pd.DataFrame) -> go.Figure:
    """Monthly late rate by shipping mode, with the overall rate dashed."""
    fig = go.Figure()
    monthly = ov.monthly_by_mode(df)
    monthly = monthly[monthly["orders"] >= ov.MIN_MONTH_ORDERS]
    for mode in ov.mode_order(df):
        part = monthly[monthly["shipping_mode"] == mode].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(part["month"]),
                y=part["late_rate"],
                name=mode,
                mode="lines",
                line={"color": MODE_COLORS.get(mode, ui_theme.SLATE), "width": 2.2},
                customdata=part["orders"],
                hovertemplate=(
                    f"{mode}<br>%{{x|%b %Y}}: %{{y:.1%}} late "
                    "(%{customdata:,} orders)<extra></extra>"
                ),
            )
        )
    overall = ov.monthly_overall(df)
    overall = overall[overall["orders"] >= ov.MIN_MONTH_ORDERS].sort_values("month")
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(overall["month"]),
            y=overall["late_rate"],
            name="All modes",
            mode="lines",
            line={"color": ui_theme.INK, "width": 1.6, "dash": "dash"},
            hovertemplate="All modes<br>%{x|%b %Y}: %{y:.1%} late<extra></extra>",
        )
    )
    fig.update_layout(
        **ui_theme.plotly_layout(
            height=CHART_HEIGHT,
            margin={**MARGIN, "b": 4},
            hovermode="closest",
            legend={
                "orientation": "h",
                "x": 0,
                "y": 1.0,
                "yanchor": "bottom",
                "font": {"size": 11},
            },
        )
    )
    fig.update_layout(margin={"l": 4, "r": 4, "t": 28, "b": 4})
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title=None)
    fig.update_xaxes(tickformat="%b %Y", title=None, showgrid=False)
    return fig


def treemap_figure(df: pd.DataFrame) -> go.Figure:
    """Late orders by market then shipping mode; color is the late rate."""
    table = ov.market_mode_table(df)
    table = table[table["late_orders"] > 0]
    by_market = table.groupby("market")[["orders", "late_orders"]].sum()
    total_late = int(by_market["late_orders"].sum())
    total_orders = int(by_market["orders"].sum())
    ids, labels, parents, values, colors, orders = (
        ["all"],
        ["All markets"],
        [""],
        [total_late],
        [ov.rate(total_late, total_orders)],
        [total_orders],
    )
    for market, row in by_market.iterrows():
        ids.append(str(market))
        labels.append(str(market))
        parents.append("all")
        values.append(int(row["late_orders"]))
        colors.append(ov.rate(row["late_orders"], row["orders"]))
        orders.append(int(row["orders"]))
    for _, row in table.iterrows():
        ids.append(f"{row['market']}|{row['shipping_mode']}")
        labels.append(str(row["shipping_mode"]))
        parents.append(str(row["market"]))
        values.append(int(row["late_orders"]))
        colors.append(float(row["late_rate"]))
        orders.append(int(row["orders"]))
    lo, hi = (min(colors), max(colors)) if colors else (0.0, 1.0)
    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            customdata=list(zip(colors, orders, strict=True)),
            marker={
                "colors": colors,
                "colorscale": RATE_SCALE,
                "cmin": lo,
                "cmax": hi,
                "colorbar": _colorbar(),
                "line": {"width": 2, "color": "#FFFFFF"},
            },
            texttemplate="<b>%{label}</b><br>%{value:,}",
            textfont={"size": 12},
            hovertemplate=(
                "%{label}<br>%{value:,} late of %{customdata[1]:,} orders"
                "<br>Late rate %{customdata[0]:.1%}<extra></extra>"
            ),
            tiling={"pad": 3},
            root={"color": "#FFFFFF"},
            pathbar={"visible": False},
        )
    )
    fig.update_layout(uniformtext={"minsize": 9, "mode": "hide"})
    fig.update_layout(**ui_theme.plotly_layout(height=CHART_HEIGHT, margin=MARGIN))
    return fig


def sankey_figure(df: pd.DataFrame) -> go.Figure:
    """Shipping mode to On time or Late, link width = orders."""
    flow = ov.mode_flow(df)
    modes = [str(m) for m in flow["shipping_mode"]]
    labels = [*modes, "On time", "Late"]
    on_idx, late_idx = len(modes), len(modes) + 1
    sources, targets, values, colors = [], [], [], []
    for i, row in flow.iterrows():
        idx = int(str(i))
        sources += [idx, idx]
        targets += [on_idx, late_idx]
        values += [int(row["on_time"]), int(row["late_orders"])]
        colors += [_rgba(ui_theme.GREEN, 0.28), _rgba(ui_theme.RED, 0.34)]
    node_colors = [MODE_COLORS.get(m, ui_theme.SLATE) for m in modes]
    node_colors += [ui_theme.GREEN, ui_theme.RED]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": node_colors,
                "pad": 18,
                "thickness": 16,
                "line": {"width": 0},
                "hovertemplate": "%{label}: %{value:,} orders<extra></extra>",
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": colors,
                "hovertemplate": (
                    "%{source.label} to %{target.label}<br>%{value:,} orders"
                    "<extra></extra>"
                ),
            },
            textfont={"size": 12, "color": ui_theme.INK},
        )
    )
    fig.update_layout(**ui_theme.plotly_layout(height=CHART_HEIGHT, margin=MARGIN))
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _on_heatmap_click(chart_key: str) -> None:
    """Turn a click on a heatmap row into the Market filter.

    Clicking the row of the market that is already applied (or deselecting
    it) clears the Market filter back to All.
    """
    state = st.session_state.get(chart_key)
    points = getattr(getattr(state, "selection", None), "points", None)
    if points is None and isinstance(state, dict):
        points = state.get("selection", {}).get("points")
    key = FILTER_KEYS["market"]
    market = points[0].get("y") if points else None
    if not market or market == st.session_state.get(key):
        st.session_state[key] = ov.ALL
    else:
        st.session_state[key] = market


def _head(title: str, caption: str | None = None) -> str:
    return f'<div class="ov-head">{ui_theme.card_heading(title, caption)}</div>'


def _filters(cube: pd.DataFrame) -> dict[str, str]:
    options = ov.filter_options(cube)
    cols = st.columns(4)
    labels = {
        "market": "Market",
        "segment": "Customer segment",
        "mode": "Shipping mode",
        "year": "Year",
    }
    chosen: dict[str, str] = {}
    for col, name in zip(cols, labels, strict=True):
        with col:
            chosen[name] = st.selectbox(
                labels[name], options[name], key=FILTER_KEYS[name]
            )
    return chosen


def _kpi_band(df: pd.DataFrame, total_orders: int) -> None:
    k = ov.kpis(df)
    share_all = ov.rate(k["orders"], total_orders)
    ui_theme.render(
        ui_theme.tiles(
            [
                ui_theme.tile(
                    "Orders",
                    f"{int(k['orders']):,}",
                    sub=f"{share_all:.0%} of all orders",
                    tip="Distinct customer orders that match the filters above.",
                ),
                ui_theme.tile(
                    "Late rate",
                    f"{k['late_rate']:.1%}",
                    tone="problem",
                    tip="Late orders divided by orders for the current filters.",
                ),
                ui_theme.tile(
                    "Late orders",
                    f"{int(k['late_orders']):,}",
                    tip="Orders that arrived after the promised date.",
                ),
                ui_theme.tile(
                    "Shipped First or Second Class",
                    f"{k['fast_share']:.1%}",
                    tone="caution",
                    tip=(
                        "Share of orders that used the two fastest paid shipping "
                        "modes, First Class or Second Class."
                    ),
                ),
            ],
            variant="four",
        )
    )


def _chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})


@st.fragment
def render_overview() -> None:
    """Render the Overview board: filter rail, KPI band and four charts."""
    cube = load_order_cube()
    chosen = _filters(cube)
    df = ov.apply_filters(cube, **chosen)
    if df.empty or int(df["orders"].sum()) == 0:
        st.info(
            f"No orders match {ov.describe(chosen)}. Choose All in one of the "
            "filters to widen the view."
        )
        return

    _kpi_band(df, int(cube["orders"].sum()))

    # A fresh chart key per filter state gives each heatmap a clean selection,
    # so every click on a market row reaches the callback.
    chart_key = f"{HEATMAP_KEY}|{'|'.join(chosen.values())}"
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            ui_theme.render(
                _head(
                    ov.heatmap_title(df),
                    "Late rate by market and shipping mode. Click a market row to filter.",
                )
            )
            st.plotly_chart(
                heatmap_figure(df),
                key=chart_key,
                on_select=lambda: _on_heatmap_click(chart_key),
                selection_mode="points",
                config={"displayModeBar": False},
            )
    with right:
        with st.container(border=True):
            ui_theme.render(
                _head(
                    ov.trend_title(df),
                    "Monthly late rate. Dashed line is all modes.",
                )
            )
            _chart(trend_figure(df), "ov_trend")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            ui_theme.render(
                _head(
                    ov.treemap_title(df),
                    "Box size is late orders. Darker is a higher late rate.",
                )
            )
            _chart(treemap_figure(df), "ov_treemap")
    with right:
        with st.container(border=True):
            ui_theme.render(
                _head(ov.sankey_title(df), "Band width is the number of orders.")
            )
            _chart(sankey_figure(df), "ov_sankey")

    ui_theme.render(
        "<div class='ov-foot'>DataCo public supply chain data, 2015 to 2018, "
        "order level. Read the full brief in the Risk predictor and Model "
        "results tabs.</div>"
    )
