"""Data explorer tab: decision box, evidence cards and a method panel."""

from collections import defaultdict
from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    late_rate_by_region_bar,
    shipping_mode_bar,
)
from app.components import ui_theme
from domain.models import Order


def _compute_late_rates_by_mode(orders: list[Order]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    late_counts: dict[str, int] = defaultdict(int)
    for order in orders:
        counts[order.shipping_mode] += 1
        if order.late_delivery_risk == 1:
            late_counts[order.shipping_mode] += 1
    return {m: late_counts[m] / counts[m] for m in counts if counts[m] > 0}


def _compute_late_rates_by_region(orders: list[Order]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    late_counts: dict[str, int] = defaultdict(int)
    for order in orders:
        counts[order.order_region] += 1
        if order.late_delivery_risk == 1:
            late_counts[order.order_region] += 1
    return {r: late_counts[r] / counts[r] for r in counts if counts[r] > 0}


def dataset_bottom_line(
    region_rates: dict[str, float], mode_rates: dict[str, float], full: bool
) -> str:
    """Decision box: does region matter, or does shipping mode?"""
    low, high = min(region_rates.values()), max(region_rates.values())
    barely = high - low <= 0.10
    sparse = not full
    if sparse:
        answer = "Too few orders per region to tell"
    elif barely:
        answer = f"Barely: every region is {low:.1%} to {high:.1%} late"
    else:
        answer = f"Yes: regions range from {low:.1%} to {high:.1%} late"
    ordered = sorted(mode_rates.items(), key=lambda kv: kv[1], reverse=True)
    picks = [o for o in ordered if o[0] in ("First Class", "Second Class")]
    if ordered[-1] not in picks:
        picks.append(ordered[-1])
    mode_text = ", ".join(f"{m.replace(' Class', '')} {r:.1%}" for m, r in picks)
    if sparse:
        sentences = [
            "The 1,000-order sample leaves only a handful of orders in each "
            "region, so regional rates are mostly noise.",
            f"Shipping mode still stands out ({mode_text}).",
        ]
        act = "Switch to the full dataset to compare regions."
    elif barely:
        sentences = [
            f"Across {len(region_rates)} regions the late rate barely moves.",
            f"Shipping mode does matter ({mode_text}).",
        ]
        act = "Focus on shipping mode, not geography."
    else:
        sentences = [
            f"Late rates differ by region, from {low:.1%} to {high:.1%}.",
            f"Shipping mode still separates orders sharply ({mode_text}).",
        ]
        act = "Check both shipping mode and region when prioritising orders."
    return ui_theme.bottom_line(
        question="Does region matter?",
        answer=answer,
        tone="neutral" if sparse else ("good" if barely else "caution"),
        sentences=sentences,
        act=act,
        confidence="High" if full else "Low",
        confidence_why=(
            "Late rates are counted directly across all orders."
            if full
            else "Only 1,000 orders, so small regions have few orders each. "
            "Switch to the full dataset for complete statistics."
        ),
        would_change="New carriers, shipping options or sales regions",
    )


def render_dataset_tab(
    pipeline: dict[str, Any],
    full_stats: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Data Explorer tab."""
    full = bool(use_full and full_stats)
    if full and full_stats:
        ds = full_stats["dataset"]
        total = ds["total_orders"]
        late_rate = ds["late_rate"]
        n_modes = len(ds["shipping_modes"])
        n_regions = len(ds["regions"])
        mode_rates = ds["late_rate_by_mode"]
        region_rates = ds["late_rate_by_region"]
        total_rows = ds.get("total_rows", total)
        note = ui_theme.source_note(
            f"Showing: Full dataset (180K rows). {total_rows:,} rows covering "
            f"{total:,} unique orders across {n_regions} regions and {n_modes} "
            "shipping modes."
        )
    else:
        orders = pipeline["orders"]
        total = len(orders)
        late = sum(1 for o in orders if o.late_delivery_risk == 1)
        late_rate = late / total if total > 0 else 0.0
        n_modes = len({o.shipping_mode for o in orders})
        n_regions = len({o.order_region for o in orders})
        mode_rates = _compute_late_rates_by_mode(orders)
        region_rates = _compute_late_rates_by_region(orders)
        note = ui_theme.source_note(
            "Showing: Sample (1K rows). A stratified sample. Switch to the full "
            "dataset in the sidebar for complete statistics.",
            warn=True,
        )

    # --- Layer 1: decision ---
    ui_theme.render(dataset_bottom_line(region_rates, mode_rates, full))
    ui_theme.render(note)
    ui_theme.render(
        ui_theme.tiles(
            [
                ui_theme.tile("Total orders", f"{total:,}"),
                ui_theme.tile(
                    "Late rate",
                    f"{late_rate:.1%}",
                    tone="problem",
                    tip="Share of orders delivered late.",
                ),
                ui_theme.tile("Shipping modes", str(n_modes)),
                ui_theme.tile("Regions", str(n_regions)),
            ]
        )
    )

    # --- Layer 2: evidence ---
    top_mode = max(mode_rates.items(), key=lambda kv: kv[1])
    with ui_theme.card(
        "Late delivery rate by shipping mode",
        f"{top_mode[0]} is the worst at {top_mode[1]:.1%} late. First Class and "
        "Second Class are treated as the high-risk shipping modes.",
    ):
        st.plotly_chart(
            ui_theme.style_figure(shipping_mode_bar(mode_rates)),
            use_container_width=True,
        )

    low, high = min(region_rates.values()), max(region_rates.values())
    if high - low <= 0.10:
        region_take = (
            f"Region barely matters: all {len(region_rates)} regions fall between "
            f"{low:.1%} and {high:.1%}, close to the {late_rate:.1%} average."
        )
    else:
        region_take = (
            f"Late rates range from {low:.1%} to {high:.1%} across "
            f"{len(region_rates)} regions (average {late_rate:.1%})."
        )
    with ui_theme.card("Late delivery rate by region", region_take):
        st.plotly_chart(
            ui_theme.style_figure(
                late_rate_by_region_bar(region_rates, overall_rate=late_rate)
            ),
            use_container_width=True,
        )

    with ui_theme.card(
        "Data integrity and leakage protection",
        "Three columns that reveal the outcome are left out, and the pipeline "
        "guards against peeking.",
        tip=(
            "Leakage means the model sees information that is only known after "
            "an order is delivered, which would make its scores look better than "
            "they are in real use."
        ),
    ):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                """
**3 columns left out so the model cannot peek at the answer:**

| Column | Reason |
|--------|--------|
| Actual shipping days | Only known after delivery |
| Delivery status | Directly states whether the order was late |
| Shipping date | Information from the future |
                """
            )

        with col_b:
            st.markdown(
                """
**Safeguards in the pipeline:**

- Orders are split into train and test sets before any encoding
- The encoder learns only from the training orders
- Property-based tests check that no information leaks
- 172 tests, 93% code coverage
                """
            )

    # --- Layer 3: method ---
    with st.expander("How we know", expanded=False):
        st.markdown(
            """
- **Metrics:** the late rate is the share of orders delivered after the promised
  date, counted per order across all lines.
- **Test design:** model results on the other tabs use a time-based split, so
  the model is tested on later orders than it learned from.
- **Leakage protection:** three columns recorded after shipping are dropped
  before any modelling.
- **Caveats:** a region that looks the same on average can still hide
  differences inside it, such as by country or product.
            """
        )

    with st.expander("Dataset source and license"):
        st.markdown(
            """
**Source:** [DataCo SMART Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Kaggle)

**Full dataset:** 180,519 rows, 65,752 unique orders, 53 columns, about 95 MB

**Class distribution:**
- Late: 54.8%
- On time: 45.2%
            """
        )
