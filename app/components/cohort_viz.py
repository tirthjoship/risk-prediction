"""Customer Segments tab — K-Means risk cohorts with data source toggle."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from adapters.visualization.labels import cluster_colors, risk_tier
from adapters.visualization.plotly_charts import cluster_scatter_2d

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def _load_full_pca() -> dict[str, Any] | None:
    """Load pre-computed PCA cluster coordinates for full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_pca_clusters.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


_TIER_COLOR_NAME = {"High": "red", "Medium": "orange", "Low": "green"}

_PCA_HELP = (
    "PCA squeezes many order details into two axes so that similar orders sit "
    "close together. The axes have no business unit of their own."
)


def _distinct_labels(cohorts: list[dict[str, Any]]) -> dict[int, str]:
    """Return display names, adding a short descriptor when names repeat.

    The descriptor comes from the region mix: a segment where one region holds
    at least 40% of orders is named after it, otherwise it is called mixed.
    """
    counts: dict[str, int] = {}
    for c in cohorts:
        counts[c["label"]] = counts.get(c["label"], 0) + 1

    names: dict[int, str] = {}
    for c in cohorts:
        name = str(c["label"])
        if counts[name] > 1:
            regions = c.get("top_regions") or {}
            top_region, share = (
                max(regions.items(), key=lambda kv: kv[1]) if regions else ("", 0.0)
            )
            if top_region and share >= 0.40:
                name = f"{name}, mostly {top_region}"
            else:
                name = f"{name}, mixed regions"
        names[int(c["cluster_id"])] = name
    return names


def _render_cohort_card(
    label: str,
    size: int,
    late_rate: float,
    avg_days: float,
    dominant_mode: str,
    top_regions: dict[str, float],
    expanded: bool = False,
) -> None:
    """Render a single cohort profile card."""
    risk_tier_name = risk_tier(late_rate)
    color = _TIER_COLOR_NAME[risk_tier_name]
    header = f":{color}[**{label}**] ({size:,} orders)"

    with st.expander(header, expanded=expanded):
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Late rate",
            f"{late_rate:.1%}",
            help="Share of these orders that arrived late.",
        )
        c2.metric(
            "Scheduled days",
            f"{avg_days:.1f}",
            help="Average number of days planned for shipping.",
        )
        c3.metric(
            "Risk level",
            risk_tier_name,
            help="High is 70% late or more, Medium is 40% to 70%, Low is under 40%.",
        )

        st.markdown(f"**Shipping mode:** {dominant_mode}")

        if top_regions:
            region_str = ", ".join(f"{r} ({v:.0%})" for r, v in top_regions.items())
            st.markdown(f"**Top regions:** {region_str}")

        if late_rate >= 0.70:
            st.error("Proactive intervention recommended")
        elif late_rate >= 0.40:
            st.warning("Monitor closely")
        else:
            st.success("Performing within tolerance")


def render_cohorts_tab(
    pipeline: dict[str, Any],
    full_stats: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Customer Segments tab."""
    st.header("Customer Segments")

    if use_full and full_stats:
        st.markdown(
            "<div class='disclaimer-box'>"
            "<strong>Showing: Full dataset (180K rows).</strong> Orders were sorted "
            "into 4 segments (65,752 unique orders) based on 10 numeric order "
            "details plus shipping mode."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='disclaimer-box warn'>"
            "<strong>Showing: Sample (1K rows).</strong> Segments are built from the "
            "1,000-order sample. Switch to the full dataset in the sidebar for the "
            "segmentation on all orders."
            "</div>",
            unsafe_allow_html=True,
        )

    if use_full and full_stats:
        cohorts_data = sorted(
            full_stats["cohorts"], key=lambda c: c["late_rate"], reverse=True
        )
        cohort_label_map = _distinct_labels(full_stats["cohorts"])
        color_map = cluster_colors(
            {int(c["cluster_id"]): c["late_rate"] for c in full_stats["cohorts"]}
        )
        name_by_id = cohort_label_map

        # Try to load pre-computed PCA
        full_pca = _load_full_pca()

        if full_pca:
            scatter_col, cards_col = st.columns([3, 2])
            with scatter_col:
                st.markdown("##### How orders group together", help=_PCA_HELP)
                st.caption(
                    f"A 2,000-order random subset of {full_pca['total_orders']:,} orders"
                )
                X_pca = np.column_stack([full_pca["pc1"], full_pca["pc2"]])
                cluster_labels = np.array(full_pca["cluster"])
                scatter_fig = cluster_scatter_2d(
                    X_pca,
                    cluster_labels,
                    cohort_labels=cohort_label_map,
                    color_map=color_map,
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
            with cards_col:
                st.markdown("##### Segment profiles (65,752 orders)")
                for c in cohorts_data:
                    _render_cohort_card(
                        label=name_by_id[int(c["cluster_id"])],
                        size=c["size"],
                        late_rate=c["late_rate"],
                        avg_days=c["avg_scheduled_days"],
                        dominant_mode=c["dominant_shipping_mode"],
                        top_regions=c.get("top_regions", {}),
                        expanded=c["late_rate"] >= 0.70,
                    )
        else:
            st.markdown("##### Segment profiles (65,752 orders)")
            for c in cohorts_data:
                _render_cohort_card(
                    label=name_by_id[int(c["cluster_id"])],
                    size=c["size"],
                    late_rate=c["late_rate"],
                    avg_days=c["avg_scheduled_days"],
                    dominant_mode=c["dominant_shipping_mode"],
                    top_regions=c.get("top_regions", {}),
                    expanded=c["late_rate"] >= 0.70,
                )
    else:
        # Sample — has live PCA scatter
        cohorts = pipeline["cohorts"]
        X_pca = pipeline["X_pca"]
        cluster_labels = pipeline["cluster_labels"]
        sample_info = [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "top_regions": c.region_distribution,
            }
            for c in cohorts
        ]
        cohort_labels = _distinct_labels(sample_info)
        sample_colors = cluster_colors({c.cluster_id: c.late_rate for c in cohorts})

        scatter_col, cards_col = st.columns([3, 2])

        with scatter_col:
            st.markdown("##### How orders group together", help=_PCA_HELP)
            scatter_fig = cluster_scatter_2d(
                X_pca,
                cluster_labels,
                cohort_labels=cohort_labels,
                color_map=sample_colors,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with cards_col:
            st.markdown("##### Segment profiles")
            sorted_cohorts = sorted(cohorts, key=lambda c: c.late_rate, reverse=True)
            for cohort in sorted_cohorts:
                top_regions = dict(
                    sorted(
                        cohort.region_distribution.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                )
                _render_cohort_card(
                    label=cohort_labels[cohort.cluster_id],
                    size=cohort.size,
                    late_rate=cohort.late_rate,
                    avg_days=cohort.avg_scheduled_days,
                    dominant_mode=cohort.dominant_shipping_mode,
                    top_regions=top_regions,
                    expanded=cohort.late_rate >= 0.70,
                )
