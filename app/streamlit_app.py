"""Streamlit app — Late Delivery Risk Prediction dashboard.

4 tabs: Risk Predictor, Model Results, Customer Segments, Data Explorer.
Sidebar toggle switches stats tabs between sample (1K) and full (180K) data.
Predict tab always uses live sample-trained model for interactivity.

Run with:
    streamlit run app/streamlit_app.py
Or:
    make app
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path so adapters/domain/application are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from adapters.data.csv_repository import DataCoCSVRepository  # noqa: E402
from adapters.ml.feature_encoder import FeatureEncoder  # noqa: E402
from adapters.ml.kmeans_clusterer import KMeansClusterer  # noqa: E402
from adapters.ml.shap_explainer import ShapExplainer  # noqa: E402
from adapters.ml.sklearn_predictor import (  # noqa: E402
    LogisticRegressionPredictor,
    XGBoostPredictor,
)
from application.use_cases import (  # noqa: E402
    ProfileClustersUseCase,
    TrainAndEvaluateUseCase,
)
from domain.models import RiskCohort  # noqa: E402
from domain.ports import ExperimentTrackerPort  # noqa: E402

# ---------------------------------------------------------------------------
# Custom CSS — tuned for light theme
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F0F2F6;
        border: 1px solid #E0E4EA;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.8; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }

    /* Disclaimer boxes */
    .disclaimer-box {
        background-color: #F7F8FA;
        border-left: 3px solid #4A9EFF;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-bottom: 1rem;
        opacity: 0.95;
    }
    .disclaimer-box.warn {
        border-left-color: #E09B3D;
    }

    /* Impact card */
    .impact-card {
        background: #F7F8FA;
        border: 1px solid #E0E4EA;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1.5rem;
    }
    .impact-card h3 { margin-top: 0; margin-bottom: 8px; }

    /* Section spacing */
    .section-gap { margin-top: 1.5rem; }

    /* Sidebar styling */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #FFFFFF;
    }
</style>
"""


# ---------------------------------------------------------------------------
# No-op tracker
# ---------------------------------------------------------------------------
class _FakeTracker(ExperimentTrackerPort):
    """No-op experiment tracker for dashboard use."""

    def start_run(self, run_name: str | None = None) -> None:
        pass

    def log_params(self, params: dict[str, str]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_model(self, model: Any, artifact_name: str) -> None:
        pass

    def log_artifact(self, obj: Any, artifact_name: str) -> None:
        pass

    def register_model(self, name: str) -> None:
        pass

    def end_run(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_full_metrics() -> dict[str, Any] | None:
    """Load pre-computed metrics from the full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_run_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_full_stats() -> dict[str, Any] | None:
    """Load pre-computed dataset stats and cohorts from the full dataset."""
    path = _PROJECT_ROOT / "data" / "metrics" / "full_dataset_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource(
    show_spinner="Training the models on the sample data. This takes about 5 seconds on first load."
)
def load_pipeline() -> dict[str, Any]:
    """Train XGBoost + LogReg on sample CSV, run clustering, compute SHAP."""
    sample_csv = _PROJECT_ROOT / "data" / "sample" / "sample.csv"

    data_repo = DataCoCSVRepository(str(sample_csv))
    orders = data_repo.get_orders()
    tracker = _FakeTracker()

    # --- XGBoost ---
    xgb_encoder = FeatureEncoder()
    xgb_model = XGBoostPredictor()
    xgb_use_case = TrainAndEvaluateUseCase(
        data_repo=data_repo,
        feature_encoder=xgb_encoder,
        model=xgb_model,
        explainer_factory=ShapExplainer,
        tracker=tracker,
    )
    xgb_result = xgb_use_case.execute(run_name="xgb-dashboard")

    # --- LogReg ---
    lr_encoder = FeatureEncoder()
    logreg_model = LogisticRegressionPredictor()
    lr_use_case = TrainAndEvaluateUseCase(
        data_repo=data_repo,
        feature_encoder=lr_encoder,
        model=logreg_model,
        explainer_factory=ShapExplainer,
        tracker=tracker,
    )
    logreg_result = lr_use_case.execute(run_name="lr-dashboard")

    # --- SHAP ---
    from domain.services import extract_features

    raw_features = [extract_features(o) for o in orders]
    X_full = xgb_encoder.transform(raw_features)
    feature_names = xgb_encoder.get_feature_names()

    xgb_explainer = ShapExplainer(xgb_model.model, feature_names)
    xgb_global = xgb_explainer.explain_global(X_full.values)

    # --- PCA + Clustering ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_full.values)

    cluster_encoder = FeatureEncoder()
    clusterer = KMeansClusterer(n_clusters=4, random_state=42)
    profile_use_case = ProfileClustersUseCase(
        data_repo=data_repo,
        feature_encoder=cluster_encoder,
        clusterer=clusterer,
    )
    cohorts: list[RiskCohort] = profile_use_case.execute()
    cluster_labels = clusterer.get_labels()
    cohort_labels_map: dict[int, str] = {c.cluster_id: c.label for c in cohorts}

    return {
        "orders": orders,
        "xgb_model": xgb_model,
        "logreg_model": logreg_model,
        "encoder": xgb_encoder,
        "xgb_result": xgb_result,
        "logreg_result": logreg_result,
        "xgb_explainer": xgb_explainer,
        "cohorts": cohorts,
        "X_pca": X_pca,
        "cluster_labels": cluster_labels,
        "cohort_labels": cohort_labels_map,
        "feature_names": feature_names,
        "X_full_encoded": X_full,
        "xgb_shap_values": xgb_global.shap_values,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Late Delivery Risk Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Settings")
        data_source = st.radio(
            "Data shown in the results tabs",
            options=["Full dataset (180K rows)", "Sample (1K rows)"],
            index=0,
            help=(
                "Model Results, Customer Segments and Data Explorer can show "
                "either the full dataset (the reported results) or a 1,000-order "
                "sample that loads faster. The Risk Predictor tab always uses a "
                "model trained on the sample so it responds instantly."
            ),
        )
        use_full = data_source == "Full dataset (180K rows)"

        st.markdown("---")
        st.markdown("### Quick facts")
        st.metric(
            "Order lines",
            "180,519",
            help="Rows in the raw file. An order can have several lines.",
        )
        st.metric("Unique orders", "65,752", help="Orders after combining their lines.")
        st.metric("Late rate", "54.8%", help="Share of orders delivered late.")
        st.metric(
            "Top signal",
            "Shipping mode",
            help="The order detail with the largest effect on the late-delivery score.",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size: 0.75rem; opacity: 0.5;'>"
            "Built by Tirth Joshi<br>"
            "<a href='https://github.com/tirthjoship/late-delivery-risk-prediction' "
            "style='color: inherit;'>View on GitHub</a>"
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Load static data first (instant, no training) ---
    full_metrics = load_full_metrics()
    full_stats = load_full_stats()

    # --- Hero (shows immediately while pipeline trains) ---
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>"
        "Late Delivery Risk Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; opacity: 0.6; margin-top: -0.5rem; "
        "margin-bottom: 1.2rem;'>"
        "Predict, explain, compare, segment</p>",
        unsafe_allow_html=True,
    )

    # --- Landing summary ---
    with st.container(border=True):
        st.subheader("The problem")
        st.markdown(
            "**More than half of orders (54.8%) arrive late.** Shipping mode is the "
            "main reason: First Class orders are late 95.3% of the time and Second "
            "Class orders 76.7% of the time, while Standard Class is late 38.1% of "
            "the time."
        )
        st.subheader(
            "What this app does",
            help=(
                "Model details: XGBoost trained on the 52,601 earliest orders and "
                "tested on the 13,151 latest ones. At a 0.5 cutoff it scores F1 = "
                "0.6648 (precision 0.7743, recall 0.5825). Shipping mode accounts for "
                "about 73% of the total feature impact (mean absolute SHAP), "
                "so the model largely re-learns a shipping-mode lookup. Experiments "
                "were tracked with MLflow and the code follows a hexagonal "
                "(ports and adapters) layout."
            ),
        )
        st.markdown(
            "It estimates the chance that an order will arrive late before it ships, "
            "shows which order details pushed that estimate up or down, and groups "
            "orders into segments. Logistics teams can use this to reroute, "
            "prioritize, or warn customers early."
        )

    # --- Technical notes expander ---
    with st.expander("How it works (technical notes)", expanded=False):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(
                """
**Data flow:**
1. The raw file passes through a leakage filter that drops 3 columns recorded after shipping.
2. Each order gets 12 features that are known before it ships.
3. Orders are split into train and test sets first, then encoded (this avoids preprocessing leakage).
4. XGBoost and Logistic Regression are compared on F1, not accuracy.
5. SHAP explains the model for all orders and for a single order.
6. MLflow logs parameters, metrics and model files.

**Key safeguard:** the model never sees actual shipping days, delivery status,
or the shipping date. Those are only known after shipment and would give away the answer.
                """
            )
        with col_b:
            st.markdown(
                """
**Tech stack:**

| Layer | Tool |
|-------|------|
| Models | XGBoost, Logistic Regression |
| Explanations | SHAP |
| Tracking | MLflow |
| Code layout | Hexagonal |
| Tests | 156 tests, 93% coverage |
| Type checks | mypy strict |
| Dashboard | Streamlit |
                """
            )

    st.markdown("---")

    # --- Train pipeline (shows animated loading while training) ---
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            "<div style='text-align: center; padding: 3rem 0;'>"
            "<h3 style='margin-bottom: 0.3rem;'>Preparing Dashboard</h3>"
            "<p style='opacity: 0.6;'>Training XGBoost + LogReg on sample data "
            "and computing explanations.<br>"
            "<em>The first load takes about 10 seconds, then it is cached.</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )
    pipeline = load_pipeline()
    loading_placeholder.empty()

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Risk Predictor",
            "Model Results",
            "Customer Segments",
            "Data Explorer",
        ]
    )

    with tab1:
        from app.components.prediction import render_prediction_tab

        render_prediction_tab(pipeline)

    with tab2:
        from app.components.model_comparison import render_model_tab

        render_model_tab(pipeline, full_metrics, use_full)

    with tab3:
        from app.components.cohort_viz import render_cohorts_tab

        render_cohorts_tab(pipeline, full_stats, use_full)

    with tab4:
        from app.components.dataset_overview import render_dataset_tab

        render_dataset_tab(pipeline, full_stats, use_full)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.4; font-size: 0.78rem;'>"
        "Built with Streamlit, XGBoost and SHAP. "
        "<a href='https://github.com/tirthjoship/late-delivery-risk-prediction' "
        "style='color: inherit;'>GitHub</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
