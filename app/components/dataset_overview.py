"""Data Explorer tab — dataset stats with data source toggle."""

from collections import defaultdict
from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    late_rate_by_region_bar,
    shipping_mode_bar,
)
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


def render_dataset_tab(
    pipeline: dict[str, Any],
    full_stats: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Data Explorer tab."""
    st.header("Data Explorer")

    if use_full and full_stats:
        ds = full_stats["dataset"]
        source_label = "Full dataset (180K rows)"
        total = ds["total_orders"]
        late = ds["late_orders"]
        late_rate = ds["late_rate"]
        n_modes = len(ds["shipping_modes"])
        n_regions = len(ds["regions"])
        mode_rates = ds["late_rate_by_mode"]
        region_rates = ds["late_rate_by_region"]
        total_rows = ds.get("total_rows", total)

        st.markdown(
            "<div class='disclaimer-box'>"
            f"<strong>Showing: {source_label}.</strong> "
            f"{total_rows:,} rows covering {total:,} unique orders across "
            f"{n_regions} regions and {n_modes} shipping modes."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        orders = pipeline["orders"]
        source_label = "Sample (1K rows)"
        total = len(orders)
        late = sum(1 for o in orders if o.late_delivery_risk == 1)
        late_rate = late / total if total > 0 else 0.0
        n_modes = len({o.shipping_mode for o in orders})
        n_regions = len({o.order_region for o in orders})
        mode_rates = _compute_late_rates_by_mode(orders)
        region_rates = _compute_late_rates_by_region(orders)

        st.markdown(
            "<div class='disclaimer-box warn'>"
            f"<strong>Showing: {source_label}.</strong> A stratified sample. "
            "Switch to the full dataset in the sidebar for complete statistics."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Key stats ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total orders", f"{total:,}")
    col2.metric("Late rate", f"{late_rate:.1%}", help="Share of orders delivered late.")
    col3.metric("Shipping modes", str(n_modes))
    col4.metric("Regions", str(n_regions))

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # --- Shipping mode ---
    st.markdown("##### Late delivery rate by shipping mode")
    st.plotly_chart(shipping_mode_bar(mode_rates), use_container_width=True)
    st.caption(
        "First Class and Second Class are treated as the high-risk shipping modes."
    )

    st.divider()

    # --- Region ---
    st.markdown("##### Late delivery rate by region")
    st.plotly_chart(
        late_rate_by_region_bar(region_rates, overall_rate=late_rate),
        use_container_width=True,
    )
    low, high = min(region_rates.values()), max(region_rates.values())
    if high - low <= 0.10:
        st.caption(
            f"Region barely matters: all {len(region_rates)} regions fall between "
            f"{low:.1%} and {high:.1%}, close to the {late_rate:.1%} average."
        )
    else:
        st.caption(
            f"Late rates range from {low:.1%} to {high:.1%} across "
            f"{len(region_rates)} regions (average {late_rate:.1%})."
        )

    st.divider()

    # --- Data integrity ---
    st.markdown(
        "##### Data integrity and leakage protection",
        help=(
            "Leakage means the model sees information that is only known after "
            "an order is delivered, which would make its scores look better than "
            "they are in real use."
        ),
    )
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
- 156 tests, 93% code coverage
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
