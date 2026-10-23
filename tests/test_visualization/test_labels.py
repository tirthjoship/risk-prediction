"""Tests for plain-language labels and risk colors."""

from adapters.visualization.labels import (
    HIGH_COLOR,
    LOW_COLOR,
    cluster_colors,
    feature_label,
    risk_tier,
)


def test_feature_label_known_and_prefixed() -> None:
    assert feature_label("sales_per_customer") == "Sales per customer"
    assert feature_label("shipping_mode_Standard Class") == "Shipping: Standard Class"
    assert feature_label("item_count") == "Items in order"
    assert feature_label("order_day_of_week") == "Order day of week"
    assert feature_label("some_new_feature") == "Some new feature"


def test_risk_tier_cutoffs() -> None:
    assert risk_tier(0.70) == "High"
    assert risk_tier(0.50) == "Medium"
    assert risk_tier(0.39) == "Low"


def test_cluster_colors_are_consistent_and_distinct_for_mediums() -> None:
    colors = cluster_colors({0: 0.51, 1: 0.38, 2: 0.80, 3: 0.53})
    assert colors[2] == HIGH_COLOR
    assert colors[1] == LOW_COLOR
    assert colors[0] != colors[3]
