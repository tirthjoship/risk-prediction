"""Plain-language labels and risk colors shared by charts and the app."""

from collections.abc import Mapping

HIGH_COLOR = "#d62728"
MEDIUM_COLORS = ("#ff7f0e", "#b8860b")
LOW_COLOR = "#2ca02c"

_FEATURE_LABELS: dict[str, str] = {
    "days_for_shipment_scheduled": "Scheduled shipment days",
    "sales_per_customer": "Sales per customer",
    "total_discount": "Total discount",
    "avg_unit_price": "Average unit price",
    "benefit_per_order": "Benefit per order",
    "total_quantity": "Total quantity",
    "order_month": "Order month",
    "order_profit_per_order": "Profit per order",
    "total_sales": "Total sales",
    "n_items": "Number of items",
}

_PREFIX_LABELS: tuple[tuple[str, str], ...] = (
    ("shipping_mode_", "Shipping: "),
    ("order_region_", "Region: "),
    ("order_country_", "Country: "),
    ("order_state_", "State: "),
)


def feature_label(name: str) -> str:
    """Return a plain label for a raw encoded feature name."""
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name]
    for prefix, text in _PREFIX_LABELS:
        if name.startswith(prefix):
            return text + name[len(prefix) :]
    return name.replace("_", " ").capitalize()


def risk_tier(late_rate: float) -> str:
    """Return High, Medium or Low for a late rate (same cutoffs everywhere)."""
    if late_rate >= 0.70:
        return "High"
    if late_rate >= 0.40:
        return "Medium"
    return "Low"


def cluster_colors(rates: Mapping[int, float]) -> dict[int, str]:
    """Map cluster id to a color: high red, medium amber shades, low green."""
    colors: dict[int, str] = {}
    medium_seen = 0
    for cluster_id in sorted(rates):
        tier = risk_tier(rates[cluster_id])
        if tier == "High":
            colors[cluster_id] = HIGH_COLOR
        elif tier == "Medium":
            colors[cluster_id] = MEDIUM_COLORS[medium_seen % len(MEDIUM_COLORS)]
            medium_seen += 1
        else:
            colors[cluster_id] = LOW_COLOR
    return colors
