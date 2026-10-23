"""Plotly visualization adapter.

Pure Plotly functions — no Streamlit imports. Each function accepts data
and returns a go.Figure. Consumed by the Streamlit app layer.
"""

import numpy as np
import plotly.graph_objects as go

from adapters.visualization.labels import (
    ACCENT_COLOR,
    HIGH_COLOR,
    LOW_COLOR,
    MEDIUM_COLORS,
    feature_label,
)


def risk_gauge(probability: float) -> go.Figure:
    """Semicircle gauge indicator for late delivery risk probability.

    Color thresholds:
      - probability > 0.70 → red (high risk)
      - probability > 0.40 → orange (medium risk)
      - probability <= 0.40 → green (low risk)

    Args:
        probability: Predicted probability in [0, 1].

    Returns:
        go.Figure with a single Indicator trace.
    """
    pct = round(probability * 100, 1)

    if probability > 0.70:
        bar_color = HIGH_COLOR
        risk_label = "HIGH RISK"
    elif probability > 0.40:
        bar_color = MEDIUM_COLORS[0]
        risk_label = "MEDIUM RISK"
    else:
        bar_color = LOW_COLOR
        risk_label = "LOW RISK"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={
                "text": f"Late delivery risk<br><span style='font-size:0.8em'>{risk_label}</span>",
                "font": {"size": 18},
            },
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#EAECEF",
                "steps": [
                    {"range": [0, 40], "color": "#E3F3E9"},
                    {"range": [40, 70], "color": "#F8EDD6"},
                    {"range": [70, 100], "color": "#F9E2E0"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.75,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        margin={"t": 80, "b": 10, "l": 20, "r": 20},
        height=280,
    )
    return fig


def confusion_matrix_heatmap(cm: tuple[tuple[int, ...], ...]) -> go.Figure:
    """Annotated heatmap for a 2x2 confusion matrix.

    Axes: Predicted (On-Time / Late) vs Actual (On-Time / Late).

    Args:
        cm: 2-tuple of 2-tuples — ((TN, FP), (FN, TP)).

    Returns:
        go.Figure with annotated Heatmap trace.
    """
    labels = ["On-Time", "Late"]
    z = [[cm[i][j] for j in range(len(cm[i]))] for i in range(len(cm))]
    text = [[f"{cm[i][j]:,}" for j in range(len(cm[i]))] for i in range(len(cm))]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#F1F4F8"], [1.0, ACCENT_COLOR]],
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis_title="Predicted by the model",
        yaxis_title="What actually happened",
        margin={"t": 20, "b": 60, "l": 110, "r": 20},
        height=300,
    )
    return fig


def shap_waterfall(
    shap_values: dict[str, float],
    expected_value: float,
) -> go.Figure:
    """Horizontal bar chart showing top-10 SHAP contributions.

    Positive SHAP values are red (push toward late), negative are blue
    (push toward on-time).

    Args:
        shap_values: Feature name → SHAP value for one prediction.
        expected_value: Base value (model average output).

    Returns:
        go.Figure with horizontal bar traces.
    """
    if not shap_values:
        fig = go.Figure()
        fig.update_layout(title="No factors to show")
        return fig

    # Sort by absolute magnitude, keep top 10
    sorted_items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_items = sorted_items[:10]

    features = [feature_label(item[0]) for item in top_items]
    values = [item[1] for item in top_items]
    colors = [HIGH_COLOR if v > 0 else ACCENT_COLOR for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Push on score: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Pushes toward on-time (left) or late (right)",
        yaxis={"autorange": "reversed"},
        margin={"t": 20, "b": 60, "l": 180, "r": 20},
        height=max(200, len(features) * 28 + 80),
    )
    return fig


def shap_importance_bar(importances: dict[str, float]) -> go.Figure:
    """Sorted horizontal bar of global SHAP feature importance.

    Args:
        importances: Feature name → mean absolute SHAP value.

    Returns:
        go.Figure with horizontal bar trace sorted by importance.
    """
    # Filter out near-zero features for cleaner display
    filtered = {k: v for k, v in importances.items() if abs(v) >= 0.005}
    if not filtered:
        filtered = importances  # fallback: show all if everything is tiny
    sorted_items = sorted(filtered.items(), key=lambda kv: kv[1])
    features = [feature_label(item[0]) for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=ACCENT_COLOR,
            hovertemplate="<b>%{y}</b><br>Average impact: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Average impact on the score (mean absolute SHAP)",
        margin={"t": 20, "b": 60, "l": 180, "r": 20},
        height=max(200, len(features) * 22 + 80),
    )
    return fig


def shap_dependence_scatter(
    feature_name: str,
    feature_values: list[float],
    shap_values: list[float],
) -> go.Figure:
    """Scatter plot of feature value vs SHAP value with color scale.

    Args:
        feature_name: Name of the feature on the x-axis.
        feature_values: Feature values for each sample.
        shap_values: Corresponding SHAP values.

    Returns:
        go.Figure with scatter trace.
    """
    fig = go.Figure(
        go.Scatter(
            x=feature_values,
            y=shap_values,
            mode="markers",
            marker={
                "color": shap_values,
                "colorscale": "RdBu_r",
                "showscale": True,
                "colorbar": {"title": "SHAP"},
                "opacity": 0.7,
                "size": 6,
            },
            hovertemplate=f"<b>{feature_name}</b>: %{{x:.3f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"SHAP Dependence: {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="SHAP value",
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
        height=350,
    )
    return fig


def cluster_scatter_2d(
    X_pca: np.ndarray,
    labels: np.ndarray,
    cohort_labels: dict[int, str] | None = None,
    color_map: dict[int, str] | None = None,
) -> go.Figure:
    """PCA scatter plot with one trace per cluster.

    Args:
        X_pca: (n_samples, 2) array of PCA-reduced coordinates.
        labels: (n_samples,) integer cluster assignments.
        cohort_labels: Optional mapping of cluster_id → display name.
        color_map: Optional mapping of cluster_id → color (e.g. by risk tier).

    Returns:
        go.Figure with one Scatter trace per cluster.
    """
    fig = go.Figure()
    unique_clusters = sorted(set(labels.tolist()))

    colors = [
        ACCENT_COLOR,
        HIGH_COLOR,
        LOW_COLOR,
        MEDIUM_COLORS[0],
        "#5C6B7A",
        MEDIUM_COLORS[1],
        "#8FA3BD",
        "#0D0F12",
    ]

    for i, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        x_vals = X_pca[mask, 0].tolist()
        y_vals = X_pca[mask, 1].tolist()

        if cohort_labels is not None:
            name = cohort_labels.get(int(cluster_id), f"Cluster {cluster_id}")
        else:
            name = f"Cluster {cluster_id}"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=name,
                marker={
                    "color": (color_map or {}).get(
                        int(cluster_id), colors[i % len(colors)]
                    ),
                    "size": 7,
                    "opacity": 0.7,
                },
                hovertemplate=f"<b>{name}</b><extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Pattern axis 1",
        yaxis_title="Pattern axis 2",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "title": {"text": "Segment"},
        },
        margin={"t": 20, "b": 40, "l": 60, "r": 20},
        height=460,
    )
    return fig


def shipping_mode_bar(data: dict[str, float]) -> go.Figure:
    """Horizontal bar of late delivery rate by shipping mode, color-coded.

    Colors:
      - rate > 0.70 → red
      - rate > 0.40 → orange
      - else → green

    Args:
        data: Shipping mode → late rate (0.0–1.0).

    Returns:
        go.Figure with horizontal bar trace.
    """
    modes = list(data.keys())
    rates = [data[m] for m in modes]
    colors = []
    for r in rates:
        if r > 0.70:
            colors.append(HIGH_COLOR)
        elif r > 0.40:
            colors.append(MEDIUM_COLORS[0])
        else:
            colors.append(LOW_COLOR)

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=modes,
            orientation="h",
            marker_color=colors,
            text=[f"{r:.1%}" for r in rates],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Late rate: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Late rate",
        xaxis_tickformat=".0%",
        xaxis={"range": [0, 1.12]},
        margin={"t": 20, "b": 50, "l": 140, "r": 40},
        height=max(200, len(modes) * 40 + 70),
    )
    return fig


def late_rate_by_region_bar(
    data: dict[str, float], overall_rate: float | None = None
) -> go.Figure:
    """Horizontal bars of late rate by region, sorted, with an average line.

    Args:
        data: Region → late rate (0.0–1.0).
        overall_rate: Overall late rate drawn as a dashed reference line.

    Returns:
        go.Figure with a horizontal bar trace.
    """
    sorted_items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    regions = [" ".join(item[0].split()) for item in sorted_items]
    rates = [item[1] for item in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=regions,
            orientation="h",
            marker_color=ACCENT_COLOR,
            text=[f"{r:.1%}" for r in rates],
            textposition="inside",
            insidetextanchor="end",
            textfont={"color": "white"},
            hovertemplate="<b>%{y}</b><br>Late rate: %{x:.1%}<extra></extra>",
        )
    )
    if overall_rate is not None:
        fig.add_vline(
            x=overall_rate,
            line_dash="dash",
            line_color="#5C6B7A",
            layer="below",
            annotation_text=f"Overall {overall_rate:.1%}",
            annotation_position="top",
        )
    fig.update_layout(
        xaxis_title="Share of orders delivered late",
        xaxis={
            "range": [0, max(0.75, max(rates) * 1.12)],
            "tickformat": ".0%",
        },
        yaxis={"autorange": "reversed"},
        margin={"t": 50, "b": 60, "l": 140, "r": 40},
        height=max(250, len(regions) * 26 + 110),
    )
    return fig
