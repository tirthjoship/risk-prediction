"""Pure data logic for the Overview tab: filters, KPI math and computed titles.

Everything here works on the order cube written by scripts/export_dashboard.py
(columns: month, year, market, shipping_mode, customer_segment, orders,
late_orders). No Streamlit or Plotly imports, so it is easy to test.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

ALL = "All"
MODE_ORDER = ["First Class", "Second Class", "Same Day", "Standard Class"]
FAST_MODES = ("First Class", "Second Class")

# A heatmap row is "the same in every market" when the markets sit within
# this many percentage points of each other.
SAME_EVERYWHERE_POINTS = 3.0
# A mode "holds its level" when its quarterly range stays within this many
# points (or the noise allowance below, if that is larger).
STEADY_POINTS = 5.0
# The all-modes line in the trend chart skips months with fewer orders.
MIN_MONTH_ORDERS = 150
# Mode lines skip months with fewer orders than this; the rate is too noisy.
MIN_PLOT_ORDERS = 30
# A mode-quarter needs this many orders to count in the trend title.
MIN_QUARTER_ORDERS = 200
# Noise allowance: this many standard errors of a quarterly late rate.
NOISE_Z = 4.5
# Heatmap cells with fewer orders than this do not count for the title.
MIN_CELL_ORDERS = 50

CUBE_COLUMNS = [
    "month",
    "year",
    "market",
    "shipping_mode",
    "customer_segment",
    "orders",
    "late_orders",
]

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def load_cube(path: Path) -> pd.DataFrame:
    """Read the committed order cube."""
    cube = pd.read_csv(path)
    missing = [c for c in CUBE_COLUMNS if c not in cube.columns]
    if missing:
        raise ValueError(f"Order cube is missing columns: {missing}")
    return cube


def filter_options(cube: pd.DataFrame) -> dict[str, list[str]]:
    """Return the choices for each filter, each starting with All."""
    modes = [m for m in MODE_ORDER if m in set(cube["shipping_mode"])]
    modes += sorted(set(cube["shipping_mode"]) - set(modes))
    return {
        "market": [ALL, *sorted(cube["market"].unique())],
        "segment": [ALL, *sorted(cube["customer_segment"].unique())],
        "mode": [ALL, *modes],
        "year": [ALL, *[str(y) for y in sorted(cube["year"].unique())]],
    }


def apply_filters(
    cube: pd.DataFrame,
    market: str = ALL,
    segment: str = ALL,
    mode: str = ALL,
    year: str = ALL,
) -> pd.DataFrame:
    """Return the cube rows that match every filter that is not All."""
    keep = pd.Series(True, index=cube.index)
    if market != ALL:
        keep &= cube["market"] == market
    if segment != ALL:
        keep &= cube["customer_segment"] == segment
    if mode != ALL:
        keep &= cube["shipping_mode"] == mode
    if year != ALL:
        keep &= cube["year"] == int(year)
    return cube[keep]


def rate(late: float, orders: float) -> float:
    """Late rate as a fraction, 0.0 when there are no orders."""
    return float(late) / float(orders) if orders else 0.0


def pct(value: float) -> str:
    """Format a fraction as a whole percent, for example 0.548 -> '55%'."""
    return f"{value * 100:.0f}%"


def kpis(df: pd.DataFrame) -> dict[str, float]:
    """Return the four headline numbers for the filtered rows."""
    orders = int(df["orders"].sum())
    late = int(df["late_orders"].sum())
    fast = int(df.loc[df["shipping_mode"].isin(FAST_MODES), "orders"].sum())
    return {
        "orders": orders,
        "late_orders": late,
        "late_rate": rate(late, orders),
        "fast_share": rate(fast, orders),
    }


def market_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    """Orders, late orders and late rate for each market and shipping mode."""
    table = df.groupby(["market", "shipping_mode"], as_index=False)[
        ["orders", "late_orders"]
    ].sum()
    table["late_rate"] = table["late_orders"] / table["orders"]
    return table


def market_order(df: pd.DataFrame) -> list[str]:
    """Markets sorted by order volume, largest first."""
    totals = df.groupby("market")["orders"].sum().sort_values(ascending=False)
    return [str(m) for m in totals.index]


def mode_order(df: pd.DataFrame) -> list[str]:
    """Shipping modes present in df, in the fixed urgency order."""
    present = set(df["shipping_mode"])
    ordered = [m for m in MODE_ORDER if m in present]
    return ordered + sorted(present - set(ordered))


def monthly_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Orders, late orders and late rate per month and shipping mode."""
    table = df.groupby(["month", "shipping_mode"], as_index=False)[
        ["orders", "late_orders"]
    ].sum()
    table["late_rate"] = table["late_orders"] / table["orders"]
    return table


def monthly_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Orders, late orders and late rate per month across all modes."""
    table = df.groupby("month", as_index=False)[["orders", "late_orders"]].sum()
    table["late_rate"] = table["late_orders"] / table["orders"]
    return table


def mode_flow(df: pd.DataFrame) -> pd.DataFrame:
    """On time and late order counts for each shipping mode (Sankey links)."""
    table = df.groupby("shipping_mode", as_index=False)[["orders", "late_orders"]].sum()
    table["on_time"] = table["orders"] - table["late_orders"]
    table["order"] = table["shipping_mode"].map(
        {m: i for i, m in enumerate(MODE_ORDER)}
    )
    return table.sort_values("order").drop(columns="order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Computed titles
# ---------------------------------------------------------------------------
def _range_text(lo: float, hi: float) -> str:
    """Whole-percent range; collapses to one number when both round alike."""
    lo_t, hi_t = pct(lo), pct(hi)
    return lo_t if lo_t == hi_t else f"{lo_t} to {hi_t}"


def heatmap_title(df: pd.DataFrame) -> str:
    """Title for the market by shipping mode heatmap.

    The "in every market" claim is made only for the shipping mode with the
    highest late rate, only when it appears in at least two markets (each with
    MIN_CELL_ORDERS orders) and only when the markets sit within
    SAME_EVERYWHERE_POINTS of each other.
    """
    if df.empty or int(df["orders"].sum()) == 0:
        return "No orders match these filters"
    table = market_mode_table(df)
    solid = table[table["orders"] >= MIN_CELL_ORDERS]
    if solid.empty:
        solid = table
    by_mode = df.groupby("shipping_mode")[["orders", "late_orders"]].sum()
    top_mode = str((by_mode["late_orders"] / by_mode["orders"]).idxmax())
    rows = solid[solid["shipping_mode"] == top_mode]
    if len(rows) >= 2:
        lo, hi = float(rows["late_rate"].min()), float(rows["late_rate"].max())
        if (hi - lo) * 100 <= SAME_EVERYWHERE_POINTS:
            return (
                f"{top_mode} is late {_range_text(lo, hi)} of the time in every market"
            )
    if len(rows) == 1 and table["market"].nunique() == 1:
        return f"{top_mode} is late {pct(float(rows['late_rate'].iloc[0]))} of the time in {rows['market'].iloc[0]}"
    lo, hi = float(solid["late_rate"].min()), float(solid["late_rate"].max())
    if len(solid) == 1:
        only = solid.iloc[0]
        return (
            f"{only['shipping_mode']} in {only['market']} is late {pct(lo)} of the time"
        )
    return f"Late rate runs from {pct(lo)} to {pct(hi)} by market and mode"


def span_text(months: int) -> str:
    """Plain-language span, for example 37 months -> 'three years'."""
    years = round(months / 12)
    if years >= 1 and abs(months - years * 12) <= 1:
        word = _NUMBER_WORDS.get(years, str(years))
        return f"{word} year" if years == 1 else f"{word} years"
    return f"{months} months"


def _complete_quarters(df: pd.DataFrame) -> set[str]:
    """Calendar quarters whose three months all appear in df."""
    months = pd.PeriodIndex(sorted(set(df["month"])), freq="M")
    counts = pd.Series(1, index=months.asfreq("Q")).groupby(level=0).sum()
    return {str(q) for q, n in counts.items() if n == 3}


def mode_quarter_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Quarterly late-rate range per mode, with a noise allowance.

    Only complete quarters count (a quarter with one month of data, such as
    2018 Q1, is left out) and only mode-quarters with at least
    MIN_QUARTER_ORDERS orders. A mode needs two such quarters. For each mode:
    lo and hi are its lowest and highest quarterly late rate, per_month is its
    median monthly orders, and tolerance is the range (in points) that random
    noise alone can produce: max(STEADY_POINTS, NOISE_Z standard errors of a
    quarterly rate at the mode's overall late rate and median quarterly orders).
    """
    complete = _complete_quarters(df)
    if not complete:
        return {}
    monthly = monthly_by_mode(df)
    monthly["quarter"] = (
        pd.PeriodIndex(monthly["month"], freq="M").asfreq("Q").astype(str)
    )
    monthly = monthly[monthly["quarter"].isin(complete)]
    quarterly = monthly.groupby(["shipping_mode", "quarter"], as_index=False)[
        ["orders", "late_orders"]
    ].sum()
    quarterly = quarterly[quarterly["orders"] >= MIN_QUARTER_ORDERS]
    stats: dict[str, dict[str, float]] = {}
    for mode, grp in quarterly.groupby("shipping_mode"):
        if len(grp) < 2:
            continue
        rates = grp["late_orders"] / grp["orders"]
        p = float(grp["late_orders"].sum() / grp["orders"].sum())
        noise = NOISE_Z * math.sqrt(p * (1 - p) / float(grp["orders"].median())) * 100
        used = monthly[
            (monthly["shipping_mode"] == mode)
            & monthly["quarter"].isin(set(grp["quarter"]))
        ]
        stats[str(mode)] = {
            "lo": float(rates.min()),
            "hi": float(rates.max()),
            "per_month": float(used["orders"].median()),
            "tolerance": max(STEADY_POINTS, noise),
            "quarters": float(len(grp)),
        }
    return stats


def _join_names(modes: list[str]) -> str:
    """'First, Second and Standard Class' style list of shipping modes."""
    if all(m.endswith(" Class") for m in modes):
        names = [m.removesuffix(" Class") for m in modes]
        suffix = " Class"
    else:
        names, suffix = list(modes), ""
    head = ", ".join(names[:-1])
    return (f"{head} and {names[-1]}" if head else names[0]) + suffix


def trend_title(df: pd.DataFrame) -> str:
    """Title for the monthly trend lines, judged on complete quarters.

    A mode "holds its level" when its quarterly late-rate range stays within
    max(STEADY_POINTS, noise allowance), so a thin mode is not called volatile
    for random noise alone. Modes outside that range are named with their
    quarterly range and typical monthly orders.
    """
    stats = mode_quarter_stats(df)
    if not stats:
        return "Not enough orders per quarter to show a trend"
    steady = [
        m for m, v in stats.items() if (v["hi"] - v["lo"]) * 100 <= v["tolerance"]
    ]
    swinging = [m for m in stats if m not in steady]
    span = span_text(3 * len(_complete_quarters(df)))
    if not swinging:
        who = "Each shipping mode holds" if len(steady) > 1 else f"{steady[0]} holds"
        return f"{who} its level for {span}"
    parts: list[str] = []
    if steady:
        verb = "hold" if len(steady) > 1 else "holds"
        parts.append(
            f"{_join_names(steady)} {verb} {'their' if len(steady) > 1 else 'its'} level for {span}"
        )
    for m in swinging:
        v = stats[m]
        parts.append(
            f"{m} ranges {pct(v['lo'])} to {pct(v['hi'])} by quarter "
            f"(about {round(v['per_month']):,} orders a month)"
        )
    return "; ".join(parts)


def largest_pool(df: pd.DataFrame) -> tuple[str, str, int] | None:
    """Market, mode and late orders of the biggest late-order pool."""
    table = market_mode_table(df)
    table = table[table["late_orders"] > 0]
    if table.empty:
        return None
    top = table.sort_values("late_orders", ascending=False).iloc[0]
    return str(top["market"]), str(top["shipping_mode"]), int(top["late_orders"])


def treemap_title(df: pd.DataFrame) -> str:
    """Title for the late-orders treemap."""
    pool = largest_pool(df)
    if pool is None:
        return "No late orders in this selection"
    market, mode, n = pool
    return f"Biggest pool of late orders: {market} {mode} ({n:,})"


def sankey_title(df: pd.DataFrame) -> str:
    """Title for the shipping mode to outcome Sankey."""
    late = int(df["late_orders"].sum())
    if late == 0:
        return "No late orders in this selection"
    fast = int(df.loc[df["shipping_mode"].isin(FAST_MODES), "late_orders"].sum())
    return f"{pct(fast / late)} of late orders shipped First or Second Class"


def describe(selection: dict[str, Any]) -> str:
    """Short text for the active filters, used in empty-state messages."""
    active = [f"{k}: {v}" for k, v in selection.items() if v != ALL]
    return ", ".join(active) if active else "all orders"
