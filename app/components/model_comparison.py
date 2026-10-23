"""Model Results tab — Strategy C with sidebar toggle.

Shows full-dataset or sample metrics based on sidebar selection.
SHAP importance always from full dataset when available.
SHAP dependence plot removed — replaced with key insight callout.
"""

from typing import Any

import streamlit as st

from adapters.visualization.plotly_charts import (
    confusion_matrix_heatmap,
    shap_importance_bar,
)

_F1_HELP = (
    "F1 balances two questions: of the orders flagged late, how many really were "
    "(precision), and of the orders that really were late, how many were caught "
    "(recall). 1.0 is perfect."
)
_AUC_HELP = (
    "How well the model ranks late orders above on-time ones. 0.5 is a coin flip "
    "and 1.0 is perfect."
)
_PRECISION_HELP = (
    "Of the orders the model flagged as late, the share that really were late."
)
_RECALL_HELP = "Of the orders that really were late, the share the model flagged."


def _source_label(use_full: bool) -> str:
    return "Full dataset (180K rows)" if use_full else "Sample (1K rows)"


def render_model_tab(
    pipeline: dict[str, Any],
    full_metrics: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Model Results tab."""
    st.header("Model Results")

    source = _source_label(use_full)
    if use_full and full_metrics:
        train_n = full_metrics.get("train_size")
        test_n = full_metrics.get("test_size")
        st.markdown(
            "<div class='disclaimer-box'>"
            f"<strong>Showing: {source}.</strong> Temporal split: the models learned "
            f"from the {train_n:,} earliest orders and were tested on the "
            f"{test_n:,} latest orders."
            "</div>",
            unsafe_allow_html=True,
        )
        lr = full_metrics["models"]["LogisticRegression"]
        xgb = full_metrics["models"]["XGBoost"]
        lr_f1, lr_auc = lr["f1"], lr["auc_roc"]
        lr_prec, lr_rec = lr["precision"], lr["recall"]
        xgb_f1, xgb_auc = xgb["f1"], xgb["auc_roc"]
        xgb_prec, xgb_rec = xgb["precision"], xgb["recall"]
        lr_cm = tuple(tuple(r) for r in lr["confusion_matrix"])
        xgb_cm = tuple(tuple(r) for r in xgb["confusion_matrix"])
        importance_data = xgb["feature_importances"]
    else:
        st.markdown(
            "<div class='disclaimer-box warn'>"
            f"<strong>Showing: {source}.</strong> The models here were trained on a "
            "1,000-order sample so the page loads quickly. Switch to the full "
            "dataset in the sidebar for the reported results."
            "</div>",
            unsafe_allow_html=True,
        )
        lr_m = pipeline["logreg_result"].metrics
        xgb_m = pipeline["xgb_result"].metrics
        lr_f1, lr_auc = lr_m.f1, lr_m.auc_roc
        lr_prec, lr_rec = lr_m.precision, lr_m.recall
        xgb_f1, xgb_auc = xgb_m.f1, xgb_m.auc_roc
        xgb_prec, xgb_rec = xgb_m.precision, xgb_m.recall
        lr_cm = lr_m.confusion_matrix
        xgb_cm = xgb_m.confusion_matrix
        importance_data = pipeline["xgb_result"].feature_importances

    # --- Metrics cards ---
    st.subheader(
        "How well the models score",
        help=(
            "Both models try to flag orders that will arrive late. Logistic "
            "Regression is the simple baseline. Differences are shown next to "
            "the XGBoost numbers."
        ),
    )
    lr_col, xgb_col = st.columns(2)

    with lr_col:
        st.markdown("#### Logistic Regression *(baseline)*")
        c1, c2 = st.columns(2)
        c1.metric("F1 score", f"{lr_f1:.4f}", help=_F1_HELP)
        c2.metric("AUC-ROC", f"{lr_auc:.4f}", help=_AUC_HELP)
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{lr_prec:.4f}", help=_PRECISION_HELP)
        c4.metric("Recall", f"{lr_rec:.4f}", help=_RECALL_HELP)

    with xgb_col:
        st.markdown("#### XGBoost *(main model)*")
        c1, c2 = st.columns(2)
        c1.metric(
            "F1 score", f"{xgb_f1:.4f}", delta=f"{xgb_f1 - lr_f1:+.4f}", help=_F1_HELP
        )
        c2.metric(
            "AUC-ROC",
            f"{xgb_auc:.4f}",
            delta=f"{xgb_auc - lr_auc:+.4f}",
            help=_AUC_HELP,
        )
        c3, c4 = st.columns(2)
        c3.metric(
            "Precision",
            f"{xgb_prec:.4f}",
            delta=f"{xgb_prec - lr_prec:+.4f}",
            help=_PRECISION_HELP,
        )
        c4.metric(
            "Recall",
            f"{xgb_rec:.4f}",
            delta=f"{xgb_rec - lr_rec:+.4f}",
            help=_RECALL_HELP,
        )

    st.divider()

    # --- Confusion Matrices ---
    st.subheader(
        "Right and wrong calls",
        help=(
            "Each box counts orders. The diagonal (on-time called on-time, late "
            "called late) is correct. The off-diagonal boxes are mistakes: late "
            "orders that were missed, and on-time orders flagged late by mistake."
        ),
    )
    cm_lr_col, cm_xgb_col = st.columns(2)
    with cm_lr_col:
        st.markdown("#### Logistic Regression")
        st.plotly_chart(confusion_matrix_heatmap(lr_cm), use_container_width=True)
    with cm_xgb_col:
        st.markdown("#### XGBoost")
        st.plotly_chart(confusion_matrix_heatmap(xgb_cm), use_container_width=True)

    st.divider()

    # --- SHAP importance ---
    st.subheader(
        "Which order details matter most",
        help=(
            "SHAP measures how much each order detail pushes the score up or "
            "down. Longer bars mean the detail has a bigger effect on the "
            "XGBoost prediction."
        ),
    )
    importance_fig = shap_importance_bar(importance_data)
    st.plotly_chart(importance_fig, use_container_width=True)

    # --- Key insight callout ---
    if use_full and full_metrics:
        st.info(
            "**Key insight:** shipping mode accounts for about 73% of the total "
            "effect on the score. The Standard Class, First Class and Second Class "
            "flags dominate, while details such as sales per customer and total "
            "discount add little. Late delivery here is mainly an operations "
            "problem, not a customer or pricing one."
        )
    else:
        st.info(
            "**Key insight:** the shipping mode flags dominate the score, while "
            "details such as sales per customer and total discount add little. "
            "Late delivery here is mainly an operations problem, not a customer "
            "or pricing one."
        )

    # --- Split comparison section ---
    if use_full and full_metrics and "random_split_comparison" in full_metrics:
        st.divider()
        st.subheader(
            "Temporal vs random split",
            help=(
                "A temporal split trains on earlier orders and tests on later "
                "ones, like real use where the future is unknown. A random split "
                "shuffles all orders before splitting. It is used here only as a "
                "stability check."
            ),
        )
        st.markdown(
            "<div class='disclaimer-box'>"
            "The temporal split above is the headline result. The random split is "
            "only a stability check: if the score stays close, the result does not "
            "depend on how the orders were divided."
            "</div>",
            unsafe_allow_html=True,
        )
        random_metrics = full_metrics["random_split_comparison"]
        if "XGBoost" in full_metrics.get("models", {}) and "XGBoost" in random_metrics:
            temporal_f1 = full_metrics["models"]["XGBoost"]["f1"]
            random_f1 = random_metrics["XGBoost"]["f1"]
            random_prec = random_metrics["XGBoost"]["precision"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F1, temporal split (headline)", f"{temporal_f1:.4f}")
            with col2:
                st.metric(
                    "F1, random split (stability check)",
                    f"{random_f1:.4f}",
                    delta=f"{random_f1 - temporal_f1:+.4f}",
                    delta_color="off",
                    help="Difference from the temporal split F1.",
                )
            with col3:
                st.metric(
                    "Precision, random split",
                    f"{random_prec:.4f}",
                    help=_PRECISION_HELP,
                )
