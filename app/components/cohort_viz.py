"""Customer segments tab: decision box, evidence cards and a method panel."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from adapters.visualization.labels import cluster_colors, risk_tier
from adapters.visualization.plotly_charts import cluster_scatter_2d
from app.components import ui_theme

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def _load_full_pca() -> dict[str, Any] | None:
    """Load pre-computed PCA cluster coordinates for full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_pca_clusters.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


_TIER_TONE = {"High": "problem", "Medium": "caution", "Low": "good"}
_TIER_ACTION = {
    "High": "Act now: reroute or warn customers",
    "Medium": "Monitor closely",
    "Low": "On track",
}

_PCA_HELP = (
    "PCA squeezes many order details into two axes so that similar orders sit "
    "close together. The axes have no business unit of their own."
)


def _distinct_labels(cohorts: list[dict[str, Any]]) -> dict[int, str]:
    """Return display names, adding a short descriptor when names repeat.

    The descriptor comes from the region mix: a segment where one region holds
    at least 40% of orders is named after it (for example "Western Europe focus"), otherwise it is called mixed.
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
                name = f"{name}, {top_region} focus"
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
    tone = _TIER_TONE[risk_tier_name]

    with st.expander(f"{label} ({size:,} orders)", expanded=expanded):
        ui_theme.render(
            ui_theme.tiles(
                [
                    ui_theme.tile(
                        "Late rate",
                        f"{late_rate:.1%}",
                        tone=tone,
                        tip="Share of these orders that arrived late.",
                    ),
                    ui_theme.tile(
                        "Scheduled days",
                        f"{avg_days:.1f}",
                        tip="Average number of days planned for shipping.",
                    ),
                ],
                variant="small",
            )
        )
        st.markdown(
            f"**Risk level:** {risk_tier_name}",
            help="High is 70% late or more, Medium is 40% to 70%, Low is under 40%.",
        )
        st.markdown(f"**Shipping mode:** {dominant_mode}")

        if top_regions:
            region_str = ", ".join(f"{r} ({v:.0%})" for r, v in top_regions.items())
            st.markdown(f"**Top regions:** {region_str}")

        ui_theme.render(ui_theme.tag(_TIER_ACTION[risk_tier_name], tone))


def segment_bottom_line(segments: list[dict[str, Any]], use_full: bool) -> str:
    """Decision box: which orders need attention first?

    Each segment dict needs size, late_rate, avg_scheduled_days and
    dominant_shipping_mode.
    """
    top = max(segments, key=lambda c: c["late_rate"])
    total = sum(c["size"] for c in segments)
    others = [c["avg_scheduled_days"] for c in segments if c is not top]
    tier = risk_tier(top["late_rate"])
    mode = top["dominant_shipping_mode"]
    chip_text = f"{mode} segment: {top['size']:,} orders at {top['late_rate']:.1%} late"
    sentences = [
        (
            f"The riskiest segment holds {top['size']:,} of {total:,} orders "
            f"({top['size'] / total:.0%}), and {top['late_rate']:.1%} of them "
            "arrive late."
        )
    ]
    if others:
        sentences.append(
            f"They are planned at {top['avg_scheduled_days']:.1f} days to ship, "
            f"against {min(others):.1f} to {max(others):.1f} days for the other "
            "segments."
        )
    if tier == "High":
        act = (
            "Prioritize orders in this segment; see Data explorer for the "
            "riskiest shipping mode (First Class)."
        )
    elif tier == "Medium":
        act = f"Monitor {mode} orders closely."
    else:
        act = "No segment needs urgent action."
    return ui_theme.bottom_line(
        question="Which orders need attention first?",
        answer=chip_text,
        tone=_TIER_TONE[tier],
        sentences=sentences,
        act=act,
        confidence="High" if use_full else "Low",
        confidence_why=(
            "Late rates are counted directly across all orders. The segment "
            "boundaries come from clustering, so they describe patterns, not causes."
            if use_full
            else "Only 1,000 orders. Switch to the full dataset for the segments "
            "on all orders."
        ),
        would_change="Re-clustering on newer orders, or new shipping options",
    )


def render_cohorts_tab(
    pipeline: dict[str, Any],
    full_stats: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Customer Segments tab."""
    full = bool(use_full and full_stats)
    total_orders = 65752

    if full and full_stats:
        segments: list[dict[str, Any]] = sorted(
            full_stats["cohorts"], key=lambda c: c["late_rate"], reverse=True
        )
        total_orders = sum(c["size"] for c in segments)
        labels = _distinct_labels(full_stats["cohorts"])
        colors = cluster_colors(
            {int(c["cluster_id"]): c["late_rate"] for c in full_stats["cohorts"]}
        )
        note = ui_theme.source_note(
            f"Showing: Full dataset (180K rows). Orders were sorted into 4 segments "
            f"({total_orders:,} unique orders) based on 10 numeric order details "
            "plus shipping mode."
        )
        pca = _load_full_pca()
    else:
        cohorts = sorted(pipeline["cohorts"], key=lambda c: c.late_rate, reverse=True)
        segments = [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "size": c.size,
                "late_rate": c.late_rate,
                "avg_scheduled_days": c.avg_scheduled_days,
                "dominant_shipping_mode": c.dominant_shipping_mode,
                "top_regions": dict(
                    sorted(
                        c.region_distribution.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                ),
            }
            for c in cohorts
        ]
        total_orders = sum(c["size"] for c in segments)
        labels = _distinct_labels(
            [
                {
                    "cluster_id": c["cluster_id"],
                    "label": c["label"],
                    "top_regions": c["top_regions"],
                }
                for c in segments
            ]
        )
        colors = cluster_colors(
            {int(c["cluster_id"]): c["late_rate"] for c in segments}
        )
        note = ui_theme.source_note(
            "Showing: Sample (1K rows). Segments are built from the 1,000-order "
            "sample. Switch to the full dataset in the sidebar for the "
            "segmentation on all orders.",
            warn=True,
        )
        pca = None

    # --- Layer 1: decision ---
    ui_theme.render(segment_bottom_line(segments, full))
    ui_theme.render(note)

    # --- Layer 2: evidence ---
    top = segments[0]
    scatter_col, cards_col = st.columns([3, 2], gap="medium")
    with (
        scatter_col,
        ui_theme.card(
            "How orders group together",
            "Each dot is an order. Red is a high-risk segment, amber medium and "
            "green low.",
            tip=_PCA_HELP,
        ),
    ):
        if pca:
            st.caption(f"A 2,000-order random subset of {pca['total_orders']:,} orders")
            x_pca = np.column_stack([pca["pc1"], pca["pc2"]])
            cluster_labels = np.array(pca["cluster"])
        else:
            x_pca = pipeline["X_pca"]
            cluster_labels = pipeline["cluster_labels"]
        st.plotly_chart(
            ui_theme.style_figure(
                cluster_scatter_2d(
                    x_pca,
                    cluster_labels,
                    cohort_labels=labels,
                    color_map=colors,
                )
            ),
            use_container_width=True,
        )
    with (
        cards_col,
        ui_theme.card(
            f"Segment profiles ({total_orders:,} orders)",
            f"{top['late_rate']:.1%} of the riskiest segment arrives late; open a "
            "segment for its profile.",
        ),
    ):
        for c in segments:
            _render_cohort_card(
                label=labels[int(c["cluster_id"])],
                size=c["size"],
                late_rate=c["late_rate"],
                avg_days=c["avg_scheduled_days"],
                dominant_mode=c["dominant_shipping_mode"],
                top_regions=c.get("top_regions", {}),
                expanded=c["late_rate"] >= 0.70,
            )

    # --- Layer 3: method ---
    with st.expander("How we know", expanded=False):
        st.markdown(
            """
- **Metrics:** each segment reports its size, its share of late orders and its
  average planned shipping time. Risk levels are High at 70% late or more,
  Medium from 40% to 70%, and Low below 40%.
- **Segment design:** K-Means clustering groups orders that look alike on 10
  numeric order details plus shipping mode. The 2D picture squeezes those
  details into two axes, so distances on it are only a rough guide.
- **Leakage protection:** only details known before shipping are used. Actual
  shipping days, delivery status and the shipping date are left out.
- **Caveats:** segments describe patterns in past orders. They do not prove what
  caused the delays, and they can shift when the data changes.
            """
        )
