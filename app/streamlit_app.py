"""Streamlit app — Late Delivery Risk Prediction dashboard.

5 tabs: Overview, Risk predictor, Model results, Customer segments, Data explorer.
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
from app.components import ui_theme  # noqa: E402
from application.use_cases import (  # noqa: E402
    ProfileClustersUseCase,
    TrainAndEvaluateUseCase,
)
from domain.models import RiskCohort  # noqa: E402
from domain.ports import ExperimentTrackerPort  # noqa: E402


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
    show_spinner="Training the models on the sample data. This takes about 10 seconds on first load."
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
# Header
# ---------------------------------------------------------------------------
def _app_header(
    full_metrics: dict[str, Any] | None, full_stats: dict[str, Any] | None
) -> str:
    """Build the app header from the saved metrics files."""
    lines, orders, late_rate = 180519, 65752, 0.548
    if full_stats:
        ds = full_stats["dataset"]
        lines = ds.get("total_rows", lines)
        orders = ds["total_orders"]
        late_rate = ds["late_rate"]
    f1_text = "F1 0.66"
    split_tip = (
        "The models learned from the earliest orders and were tested on the "
        "latest ones, like real use where the future is unknown."
    )
    f1_tip = (
        "F1 balances precision and recall for XGBoost on the later orders, at a "
        "0.5 cutoff. 1.0 is perfect."
    )
    if full_metrics:
        xgb = full_metrics["models"]["XGBoost"]
        f1_text = f"F1 {xgb['f1']:.2f}"
        split_tip = (
            f"The models learned from the {full_metrics['train_size']:,} earliest "
            f"orders and were tested on the {full_metrics['test_size']:,} latest "
            "ones, like real use where the future is unknown."
        )
        f1_tip = (
            f"XGBoost on the later orders at a 0.5 cutoff: F1 {xgb['f1']:.4f} "
            f"(precision {xgb['precision']:.4f}, recall {xgb['recall']:.4f}). "
            "1.0 is perfect."
        )
    return ui_theme.header(
        eyebrow="SUPPLY CHAIN \u00b7 LATE DELIVERY RISK",
        headline="Which orders will arrive late, and why?",
        summary=(
            f"{late_rate:.1%} of orders arrive late. This brief scores each order "
            "before it ships, shows what drives the risk, and says what to do first."
        ),
        chips=[
            (f"{lines // 1000}K order lines", None),
            (f"{orders:,} orders", None),
            ("Time-based test", split_tip),
            (f1_text, f1_tip),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Late Delivery Risk Prediction",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ui_theme.inject_css()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Settings")
        data_source = st.radio(
            "Data shown in the results tabs",
            options=["Full dataset (180K rows)", "Sample (1K rows)"],
            index=0,
            help=(
                "Model results, Customer segments and Data explorer can show "
                "either the full dataset (the reported results) or a 1,000-order "
                "sample that loads faster. The Risk predictor tab always uses a "
                "model trained on the sample so it responds instantly."
            ),
        )
        use_full = data_source == "Full dataset (180K rows)"

        st.markdown("---")
        ui_theme.render(ui_theme.side_title("Quick facts"))
        ui_theme.render(
            ui_theme.tiles(
                [
                    ui_theme.tile(
                        "Order lines",
                        "180,519",
                        tip="Rows in the raw file. An order can have several lines.",
                    ),
                    ui_theme.tile(
                        "Unique orders",
                        "65,752",
                        tip="Orders after combining their lines.",
                    ),
                    ui_theme.tile(
                        "Late rate",
                        "54.8%",
                        tone="problem",
                        tip="Share of orders delivered late.",
                    ),
                    ui_theme.tile(
                        "Top signal",
                        "Shipping mode",
                        tip=(
                            "The order detail with the largest effect on the "
                            "late-delivery score."
                        ),
                    ),
                ],
                variant="small stack",
            )
        )

        st.markdown("---")
        ui_theme.render(
            "<div class='ld-side-foot'>Built by Tirth Joshi<br>"
            "<a href='https://github.com/tirthjoship/risk-prediction'>"
            "View on GitHub</a></div>"
        )

    # --- Load static data first (instant, no training) ---
    full_metrics = load_full_metrics()
    full_stats = load_full_stats()

    # --- Header (shows immediately while the pipeline trains) ---
    ui_theme.render(_app_header(full_metrics, full_stats))

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
| Tests | 172 tests, 93% coverage |
| Type checks | mypy strict |
| Dashboard | Streamlit |
                """
            )

    # --- Tabs (Overview renders first, from the committed cube, no training) ---
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Risk predictor",
            "Model results",
            "Customer segments",
            "Data explorer",
        ]
    )

    with tab0:
        from app.components.overview import render_overview

        render_overview()

    # --- Train pipeline (shows a loading note inside the Risk predictor tab) ---
    with tab1:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown(
                "<div style='text-align: center; padding: 3rem 0;'>"
                "<div class='ld-card-title'>Preparing the dashboard</div>"
                "<p class='ld-takeaway'>Training XGBoost and logistic regression on "
                "the sample data and computing explanations.<br>"
                "The first load takes about 10 seconds, then it is cached.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        pipeline = load_pipeline()
        loading_placeholder.empty()

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
    ui_theme.render(
        "<div class='ld-side-foot' style='text-align: center;'>"
        "Built with Streamlit, XGBoost and SHAP. "
        "<a href='https://github.com/tirthjoship/risk-prediction'>"
        "GitHub</a></div>"
    )


if __name__ == "__main__":
    main()
