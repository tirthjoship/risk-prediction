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


def render_prediction_tab(pipeline: dict[str, Any]) -> None:
    """Render the Predict & Explain tab."""
    st.header("Predict Late Delivery Risk")
    st.caption(
        "Scores come from a model trained on a 1,000-order sample so they update "
        "instantly. See the Model Results tab for results on the full dataset.",
        help=(
            "The sample model shows how the predictor behaves. Its scores are not "
            "the reported accuracy, which comes from the full 65,752-order dataset."
        ),
    )

    left_col, spacer, right_col = st.columns([4, 1, 6])

    with left_col:
        st.markdown("##### Order Details")

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
            options=[
                "Central America",
                "Eastern Europe",
                "North America",
                "Oceania",
                "South America",
                "Southeast Asia",
                "Southern Africa",
                "Western Europe",
            ],
            index=2,
        )

        c1, c2 = st.columns(2)
        with c1:
            sales = st.number_input(
                "Sales ($)",
                min_value=0.0,
                max_value=10000.0,
                value=200.0,
                step=10.0,
                help="Total value of the order in dollars.",
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
        with c2:
            profit = st.number_input(
                "Profit ($)",
                min_value=-1000.0,
                max_value=5000.0,
                value=30.0,
                step=5.0,
                help="Profit on the whole order in dollars. Negative means a loss.",
            )

        st.markdown("")
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

            # Risk gauge
            gauge_fig = risk_gauge(result.probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk label badge
            if result.risk_label:
                st.error(
                    f"**Late delivery predicted:** {result.probability:.1%} chance of arriving late"
                )
            else:
                st.success(
                    f"**On-time delivery expected:** {result.probability:.1%} chance of arriving late"
                )

            # SHAP waterfall
            st.markdown(
                "##### What drove this prediction?",
                help=(
                    "Each bar shows how much one order detail pushed the score "
                    "toward late (right) or toward on-time (left), compared with "
                    "the average order."
                ),
            )
            local_result = explainer.explain_local(
                encoder.transform([extract_features(order)]).values,
                index=0,
            )
            waterfall_fig = shap_waterfall(
                local_result.shap_values, local_result.expected_value
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

            # Top 3 contributing factors
            sorted_shap = sorted(
                local_result.shap_values.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            st.markdown("##### Top contributing factors")
            for feature, value in sorted_shap[:3]:
                direction = "raises" if value > 0 else "lowers"
                st.markdown(
                    f"**{feature_label(feature)}** {direction} the late risk "
                    f"(SHAP {value:+.2f})"
                )
        else:
            st.markdown("")
            st.markdown("")
            st.info(
                "Fill in the order details on the left and click **Predict risk** "
                "to get a risk score with an explanation."
            )
            st.markdown("")

            with st.expander("How to read the results", expanded=True):
                st.markdown(
                    """
**Risk gauge:**
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
