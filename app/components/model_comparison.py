"""Model results tab: decision box, evidence cards and a method panel.

Shows full-dataset or sample metrics based on the sidebar selection.
"""

from typing import Any

import streamlit as st

from adapters.visualization.labels import feature_label
from adapters.visualization.plotly_charts import (
    confusion_matrix_heatmap,
    shap_importance_bar,
)
from app.components import ui_theme

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


# Share of the total feature impact (mean absolute SHAP) that sits in the
# shipping-mode flags, as reported with the full-dataset run.
_SHIPPING_SHAP_PCT = 73
_BARELY_F1_GAP = 0.02


def _source_label(use_full: bool) -> str:
    return "Full dataset (180K rows)" if use_full else "Sample (1K rows)"


def model_bottom_line(lr_f1: float, xgb_f1: float, use_full: bool) -> str:
    """Decision box: does a complex model beat a simple rule?"""
    gap = xgb_f1 - lr_f1
    trails = gap <= 0
    barely = gap < _BARELY_F1_GAP
    if trails:
        answer = f"No: {gap:+.3f} F1"
    elif barely:
        answer = f"Barely: {gap:+.3f} F1"
    else:
        answer = f"Yes: {gap:+.3f} F1"
    test_name = "the time-based test" if use_full else "this 1,000-order sample"
    sentences = [
        (
            f"XGBoost scores F1 {xgb_f1:.2f} against {lr_f1:.2f} for logistic "
            f"regression on {test_name}."
        ),
    ]
    if use_full:
        sentences.append(
            f"Shipping mode explains most of the signal ({_SHIPPING_SHAP_PCT}% "
            "of SHAP), so the model largely re-learns a shipping-mode lookup."
        )
    else:
        sentences.append(
            "The shipping-mode flags dominate the score, so the model largely "
            "re-learns a shipping-mode lookup."
        )
    return ui_theme.bottom_line(
        question="Does a complex model beat a simple rule?",
        answer=answer,
        tone="caution" if barely else "good",
        sentences=sentences,
        act=(
            "Use the shipping-mode rule; keep the model only for ranking."
            if barely
            else "Use the model, and keep the simple rule as a baseline check."
        ),
        confidence="High" if use_full else "Low",
        confidence_why=(
            "Tested on later orders the model never saw, and a random split "
            "gives a similar score."
            if use_full
            else "Only 1,000 orders. Switch to the full dataset for the "
            "reported results."
        ),
        would_change="New carriers or shipping options",
    )


def _metric_tiles(
    f1: float,
    auc: float,
    prec: float,
    rec: float,
    base: tuple[float, float, float, float] | None,
) -> str:
    """Four big-number tiles; deltas against the baseline when given."""
    values = (f1, auc, prec, rec)
    labels = ("F1 score", "AUC-ROC", "Precision", "Recall")
    tips = (_F1_HELP, _AUC_HELP, _PRECISION_HELP, _RECALL_HELP)
    items = []
    for i, (label, value, tip) in enumerate(zip(labels, values, tips, strict=True)):
        sub = f"{value - base[i]:+.4f} vs baseline" if base else "baseline"
        items.append(ui_theme.tile(label, f"{value:.4f}", sub=sub, tip=tip))
    return ui_theme.tiles(items, variant="small")


def render_model_tab(
    pipeline: dict[str, Any],
    full_metrics: dict[str, Any] | None = None,
    use_full: bool = True,
) -> None:
    """Render the Model Results tab."""
    source = _source_label(use_full)
    if use_full and full_metrics:
        train_n = full_metrics.get("train_size")
        test_n = full_metrics.get("test_size")
        note = ui_theme.source_note(
            f"Showing: {source}. Time-based test: the models learned from the "
            f"{train_n:,} earliest orders and were tested on the {test_n:,} "
            "latest orders."
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
        note = ui_theme.source_note(
            f"Showing: {source}. The models here were trained on a 1,000-order "
            "sample so the page loads quickly. Switch to the full dataset in the "
            "sidebar for the reported results.",
            warn=True,
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

    # --- Layer 1: decision ---
    ui_theme.render(model_bottom_line(lr_f1, xgb_f1, use_full and bool(full_metrics)))
    ui_theme.render(note)

    # --- Layer 2: evidence ---
    with ui_theme.card(
        "How well the models score",
        (
            f"XGBoost trails the baseline by {abs(xgb_f1 - lr_f1):.4f} F1."
            if xgb_f1 - lr_f1 <= 0
            else (
                f"XGBoost beats the baseline by {xgb_f1 - lr_f1:+.4f} F1, a small "
                "gap next to the shipping-mode signal."
                if xgb_f1 - lr_f1 < _BARELY_F1_GAP
                else f"XGBoost leads the baseline by {xgb_f1 - lr_f1:+.4f} F1."
            )
        ),
        tip=(
            "Both models try to flag orders that will arrive late. Logistic "
            "Regression is the simple baseline. Differences are shown under "
            "the XGBoost numbers."
        ),
    ):
        lr_col, xgb_col = st.columns(2)
        with lr_col:
            st.markdown("**Logistic regression** *(baseline)*")
            ui_theme.render(_metric_tiles(lr_f1, lr_auc, lr_prec, lr_rec, None))
        with xgb_col:
            st.markdown("**XGBoost** *(main model)*")
            ui_theme.render(
                _metric_tiles(
                    xgb_f1,
                    xgb_auc,
                    xgb_prec,
                    xgb_rec,
                    (lr_f1, lr_auc, lr_prec, lr_rec),
                )
            )

    lr_tp = lr_cm[1][1]
    xgb_tp = xgb_cm[1][1]
    with ui_theme.card(
        "Right and wrong calls",
        f"XGBoost correctly flags {xgb_tp:,} late orders; logistic regression "
        f"correctly flags {lr_tp:,}.",
        tip=(
            "Each box counts orders. The diagonal (on-time called on-time, late "
            "called late) is correct. The off-diagonal boxes are mistakes: late "
            "orders that were missed, and on-time orders flagged late by mistake."
        ),
    ):
        cm_lr_col, cm_xgb_col = st.columns(2)
        with cm_lr_col:
            st.markdown("**Logistic regression**")
            st.plotly_chart(
                ui_theme.style_figure(confusion_matrix_heatmap(lr_cm)),
                use_container_width=True,
            )
        with cm_xgb_col:
            st.markdown("**XGBoost**")
            st.plotly_chart(
                ui_theme.style_figure(confusion_matrix_heatmap(xgb_cm)),
                use_container_width=True,
            )

    top_feature = feature_label(max(importance_data.items(), key=lambda kv: kv[1])[0])
    if use_full and full_metrics:
        shap_take = (
            f"Shipping mode accounts for about {_SHIPPING_SHAP_PCT}% of the total "
            "effect on the score, while details such as sales per customer and "
            "total discount add little. Late delivery here is mainly an "
            "operations problem, not a customer or pricing one."
        )
    else:
        shap_take = (
            f"The shipping mode flags dominate (largest bar: {top_feature}), "
            "while details such as sales per customer and total discount add "
            "little."
        )
    with ui_theme.card(
        "Which order details matter most",
        shap_take,
        tip=(
            "SHAP measures how much each order detail pushes the score up or "
            "down. Longer bars mean the detail has a bigger effect on the "
            "XGBoost prediction."
        ),
    ):
        st.plotly_chart(
            ui_theme.style_figure(shap_importance_bar(importance_data)),
            use_container_width=True,
        )

    if use_full and full_metrics and "random_split_comparison" in full_metrics:
        random_metrics = full_metrics["random_split_comparison"]
        if "XGBoost" in full_metrics.get("models", {}) and "XGBoost" in random_metrics:
            temporal_f1 = full_metrics["models"]["XGBoost"]["f1"]
            random_f1 = random_metrics["XGBoost"]["f1"]
            random_prec = random_metrics["XGBoost"]["precision"]
            with ui_theme.card(
                "Time-based vs random split",
                "The time-based split is the headline result. The random split is "
                f"only a stability check, and it lands {random_f1 - temporal_f1:+.4f} "
                "F1 away.",
                tip=(
                    "A time-based split trains on earlier orders and tests on "
                    "later ones, like real use where the future is unknown. A "
                    "random split shuffles all orders before splitting. It is "
                    "used here only as a stability check."
                ),
            ):
                ui_theme.render(
                    ui_theme.tiles(
                        [
                            ui_theme.tile(
                                "F1, time-based split (headline)",
                                f"{temporal_f1:.4f}",
                                tip=_F1_HELP,
                            ),
                            ui_theme.tile(
                                "F1, random split (stability check)",
                                f"{random_f1:.4f}",
                                sub=f"{random_f1 - temporal_f1:+.4f} vs time-based",
                                tip="Difference from the time-based split F1.",
                            ),
                            ui_theme.tile(
                                "Precision, random split",
                                f"{random_prec:.4f}",
                                tip=_PRECISION_HELP,
                            ),
                        ]
                    )
                )

    # --- Layer 3: method ---
    with st.expander("How we know", expanded=False):
        st.markdown(
            """
- **Metrics:** F1 balances precision and recall, so a model cannot look good by
  predicting one answer. AUC-ROC checks how well late orders are ranked above
  on-time ones. Accuracy is not used.
- **Test design:** orders are sorted by date. The models learn from the earliest
  orders and are scored on the latest ones, like real use where the future is
  unknown. A random split is shown only as a stability check.
- **Leakage protection:** actual shipping days, delivery status and the shipping
  date are dropped, and the train and test split happens before any encoding.
- **Caveats:** the conclusion rests on one data set from one company. New
  carriers or new shipping options could change which model is better.
            """
        )
