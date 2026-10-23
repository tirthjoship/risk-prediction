"""Prediction & Explanation tab component.

Renders an order input form on the left and a risk gauge + SHAP waterfall
on the right when the user clicks "Predict Risk".
"""

from datetime import datetime
from typing import Any

import streamlit as st

from adapters.ml.shap_explainer import ShapExplainer
from adapters.visualization.labels import feature_label
from adapters.visualization.plotly_charts import risk_gauge, shap_waterfall
from app.components import ui_theme
from application.use_cases import PredictSingleOrderUseCase
from domain.models import Order, OrderItem
from domain.services import extract_features


def _build_order_from_form(
    shipping_mode: str,
    days: int,
    region: str,
    benefit: float,
    sales: float,
    profit: float,
) -> Order:
    """Construct a minimal Order from dashboard form inputs."""
    item = OrderItem(
        order_item_id=1,
        product_card_id="DASH-0001",
        quantity=1,
        unit_price=sales,
        discount=0.0,
        discount_rate=0.0,
        profit_ratio=profit / sales if sales > 0 else 0.0,
        sales=sales,
        item_total=sales,
    )
    return Order(
        order_id=0,
        order_date=datetime.now(),
        order_customer_id=0,
        order_region=region,
        order_country="USA",
        order_state="CA",
        order_status="Complete",
        order_city="San Francisco",
        order_zipcode="00000",
        shipping_mode=shipping_mode,
        days_for_shipment_scheduled=days,
        benefit_per_order=benefit,
        sales_per_customer=sales,
        order_profit_per_order=profit,
        items=[item],
        late_delivery_risk=None,
    )


_ACT_BY_BAND = {
    "High": "Reroute or warn the customer now",
    "Medium": "Monitor and prepare a backup carrier",
    "Low": "Proceed normally",
}


def _band(probability: float) -> str:
    """Return High, Medium or Low using the same cutoffs as the gauge."""
    if probability > 0.70:
        return "High"
    if probability > 0.40:
        return "Medium"
    return "Low"


def _prompt_bottom_line() -> str:
    """Neutral decision box shown before the first prediction."""
    return ui_theme.bottom_line(
        question="Will this order arrive late?",
        answer="Enter an order and click Predict risk",
        tone="neutral",
        sentences=[
            (
                "Fill in the order details, then click Predict risk. The answer, "
                "the reason and the suggested action appear here."
            )
        ],
    )


def _result_bottom_line(
    probability: float, shipping_mode: str, days: int, top_factor: str, raises: bool
) -> str:
    """Decision box for one scored order."""
    band = _band(probability)
    direction = "pushes the risk up" if raises else "pushes the risk down"
    return ui_theme.bottom_line(
        question="Will this order arrive late?",
        answer=f"{band} risk: {probability:.1%} chance of arriving late",
        tone=ui_theme.tone_for_probability(probability),
        sentences=[
            (
                f"For a {shipping_mode} order planned to ship in {days} "
                f"{'day' if days == 1 else 'days'}, the model puts the chance of "
                f"a late delivery at {probability:.1%}."
            ),
            f"The biggest single factor is {top_factor}, which {direction}.",
        ],
        act=_ACT_BY_BAND[band],
        confidence="Indicative",
        confidence_why=(
            "Scores come from a model trained on a 1,000-order sample so they "
            "update instantly. The reported accuracy is on the Model results tab."
        ),
        would_change="A different shipping mode or planned shipping time",
    )


def render_prediction_tab(pipeline: dict[str, Any]) -> None:
    """Render the Predict & Explain tab."""
    bottom_slot = st.empty()
    bottom_slot.markdown(_prompt_bottom_line(), unsafe_allow_html=True)

    region_options = sorted({o.order_region for o in pipeline["orders"]})

    left_col, right_col = st.columns([4, 6], gap="large")

    with (
        left_col,
        ui_theme.card(
            "Order details",
            "Describe one order before it ships.",
            tip=(
                "Scores come from a model trained on a 1,000-order sample so they "
                "update instantly. The reported accuracy comes from the full "
                "65,752-order dataset."
            ),
        ),
    ):
        shipping_mode = st.selectbox(
            "Shipping mode",
            options=["Standard Class", "First Class", "Second Class", "Same Day"],
            index=0,
            help=(
                "First Class orders are late about 95% of the time and Second "
                "Class orders about 77% of the time, so both are high risk."
            ),
        )

        days = st.slider(
            "Scheduled shipment days",
            min_value=1,
            max_value=10,
            value=4,
            help="Days the seller planned to take to ship the order.",
        )

        region = st.selectbox(
            "Order region",
            options=region_options,
            index=(
                region_options.index("Western Europe")
                if "Western Europe" in region_options
                else 0
            ),
            format_func=lambda r: " ".join(r.split()),
        )

        c1, c2 = st.columns(2)
        sales = c1.number_input(
            "Sales ($)",
            min_value=0.0,
            max_value=10000.0,
            value=200.0,
            step=10.0,
            help="Total value of the order in dollars.",
        )
        profit = c2.number_input(
            "Profit ($)",
            min_value=-1000.0,
            max_value=5000.0,
            value=30.0,
            step=5.0,
            help="Profit on the whole order in dollars. Negative means a loss.",
        )
        benefit = st.number_input(
            "Benefit ($)",
            min_value=-500.0,
            max_value=2000.0,
            value=50.0,
            step=5.0,
            help=(
                "Benefit per order: the money the order is expected to earn "
                "for the company. Use a negative number for an expected loss."
            ),
        )

        predict_clicked = st.button(
            "Predict risk", type="primary", use_container_width=True
        )

    with right_col:
        if predict_clicked:
            order = _build_order_from_form(
                shipping_mode=shipping_mode,
                days=days,
                region=region,
                benefit=benefit,
                sales=sales,
                profit=profit,
            )

            encoder = pipeline["encoder"]
            model = pipeline["xgb_model"]
            feature_names = pipeline["feature_names"]
            explainer = ShapExplainer(model.model, feature_names)

            use_case = PredictSingleOrderUseCase(
                feature_encoder=encoder,
                model=model,
                explainer=explainer,
            )
            result = use_case.execute(order)

            local_result = explainer.explain_local(
                encoder.transform([extract_features(order)]).values,
                index=0,
            )
            sorted_shap = sorted(
                local_result.shap_values.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            top_name, top_value = sorted_shap[0]
            bottom_slot.markdown(
                _result_bottom_line(
                    result.probability,
                    shipping_mode,
                    days,
                    feature_label(top_name),
                    top_value > 0,
                ),
                unsafe_allow_html=True,
            )

            outcome_help = (
                "The bands are High above 70%, Medium from 40 to 70% and Low "
                "below 40%. The model itself flags an order as late at a 0.35 "
                "cutoff, which is the operating point chosen to catch more late "
                "orders."
            )
            with ui_theme.card(
                "Risk score",
                f"{result.probability:.1%} chance of arriving late, in the "
                f"{_band(result.probability).lower()} band.",
                tip=outcome_help,
            ):
                st.plotly_chart(
                    ui_theme.style_figure(risk_gauge(result.probability)),
                    use_container_width=True,
                )

            with ui_theme.card(
                "What drove this prediction?",
                f"{feature_label(top_name)} had the largest push, "
                f"{'toward late' if top_value > 0 else 'toward on-time'}.",
                tip=(
                    "Each bar shows how much one order detail pushed the score "
                    "toward late (right) or toward on-time (left), compared with "
                    "the average order. The starting point for an average order "
                    f"is {local_result.expected_value:.3f} (SHAP base value)."
                ),
            ):
                st.plotly_chart(
                    ui_theme.style_figure(
                        shap_waterfall(
                            local_result.shap_values, local_result.expected_value
                        )
                    ),
                    use_container_width=True,
                )

            with ui_theme.card(
                "Top contributing factors",
                "The three order details that moved this score the most.",
            ):
                for feature, value in sorted_shap[:3]:
                    direction = "raises" if value > 0 else "lowers"
                    st.markdown(
                        f"**{feature_label(feature)}** {direction} the late risk",
                        help=f"SHAP value: {value:+.2f}",
                    )
        else:
            with ui_theme.card(
                "How to read the results",
                "A risk band, a gauge and the order details that drove the score.",
            ):
                st.markdown(
                    """
**Risk bands:**
- :red[**High**] (above 70%): reroute the order or notify the customer
- :orange[**Medium**] (40 to 70%): monitor closely
- :green[**Low**] (below 40%): proceed normally

**Factor chart:**
Each bar shows how an order detail pushed the score toward late
or toward on-time, compared with the average order.

**Try this:** change Shipping mode from Standard Class to First Class
and watch the gauge jump. Shipping mode is the strongest signal.
                    """
                )

    with st.expander("How we know", expanded=False):
        st.markdown(
            """
- **Model:** XGBoost trained on a 1,000-order sample, so scores update instantly.
  The reported accuracy (F1 0.66) comes from the full dataset on the Model results tab.
- **Test design:** the full-data results use a time-based split. The model learns
  from earlier orders and is tested on later ones.
- **Leakage protection:** the model never sees actual shipping days, delivery
  status or the shipping date, because those are only known after shipment.
- **Explanations:** SHAP shows how each order detail moved the score compared
  with the average order.
- **Caveats:** the input form fills a few fields with fixed defaults (for example
  country and city), so treat scores as a guide to how the model behaves, not as a
  forecast for a specific shipment.
            """
        )
