"""Generate model_evaluation_deep_dive.ipynb via nbformat.

Run: python scripts/generate_notebook.py
Output: notebooks/model_evaluation_deep_dive.ipynb
"""

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "model_evaluation_deep_dive.ipynb"

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = []

# --- Section 0: Title + Setup ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """# Model Evaluation Deep Dive

> **Purpose:** Critically evaluate the late delivery risk model beyond standard metrics.
> Temporal validation, calibration, cost-sensitive thresholds, error analysis,
> and point-in-time feature engineering.

## Key Questions
1. Does random split overestimate performance vs temporal split?
2. Are predicted probabilities well-calibrated?
3. What threshold minimizes business cost (FN=3x FP)?
4. Where does the model systematically fail?
5. Can customer history features (point-in-time) improve F1?
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Project imports
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapters.data.csv_repository import DataCoCSVRepository
from adapters.ml.feature_encoder import FeatureEncoder
from adapters.ml.sklearn_predictor import XGBoostPredictor, LogisticRegressionPredictor
from domain.services import extract_features

plt.style.use("seaborn-v0_8-darkgrid")

DATA_PATH = Path("..") / "data" / "raw" / "DataCoSupplyChainDataset.csv"  # relative to notebooks/
print(f"Loading data from {DATA_PATH}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Load orders
repo = DataCoCSVRepository(DATA_PATH)
orders = repo.get_orders()
print(f"Loaded {len(orders):,} orders")
print(f"Date range: {min(o.order_date for o in orders):%Y-%m-%d} to {max(o.order_date for o in orders):%Y-%m-%d}")
print(f"Late delivery rate: {sum(o.late_delivery_risk for o in orders) / len(orders):.1%}")
"""
    )
)

# --- Section 1: Temporal vs Random Split ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 1. Temporal vs Random Split Comparison

**Hypothesis:** Random split leaks future distribution information into training.
Temporal split (train on past, test on future) gives more realistic performance estimate.
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Extract features and labels
raw_features = [extract_features(o) for o in orders]
labels = np.array([o.late_delivery_risk for o in orders])

# --- Random Split (current approach) ---
raw_train_r, raw_test_r, y_train_r, y_test_r = train_test_split(
    raw_features, labels, test_size=0.2, random_state=42, stratify=labels
)

encoder_r = FeatureEncoder()
X_train_r = encoder_r.fit_transform(raw_train_r)
X_test_r = encoder_r.transform(raw_test_r)

model_r = XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
model_r.train(X_train_r.values, y_train_r)
y_proba_r = model_r.predict_proba(X_test_r.values)
y_pred_r = (y_proba_r >= 0.5).astype(int)

print("=== Random Split ===")
print(f"F1:        {f1_score(y_test_r, y_pred_r):.4f}")
print(f"Precision: {precision_score(y_test_r, y_pred_r):.4f}")
print(f"Recall:    {recall_score(y_test_r, y_pred_r):.4f}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# --- Temporal Split ---
sorted_indices = sorted(range(len(orders)), key=lambda i: orders[i].order_date)
cutoff = int(len(sorted_indices) * 0.8)
train_idx = sorted_indices[:cutoff]
test_idx = sorted_indices[cutoff:]

raw_train_t = [raw_features[i] for i in train_idx]
raw_test_t = [raw_features[i] for i in test_idx]
y_train_t = labels[train_idx]
y_test_t = labels[test_idx]

encoder_t = FeatureEncoder()
X_train_t = encoder_t.fit_transform(raw_train_t)
X_test_t = encoder_t.transform(raw_test_t)

model_t = XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
model_t.train(X_train_t.values, y_train_t)
y_proba_t = model_t.predict_proba(X_test_t.values)
y_pred_t = (y_proba_t >= 0.5).astype(int)

print("=== Temporal Split ===")
print(f"F1:        {f1_score(y_test_t, y_pred_t):.4f}")
print(f"Precision: {precision_score(y_test_t, y_pred_t):.4f}")
print(f"Recall:    {recall_score(y_test_t, y_pred_t):.4f}")
print(f"\\nTrain period: {orders[train_idx[0]].order_date:%Y-%m-%d} to {orders[train_idx[-1]].order_date:%Y-%m-%d}")
print(f"Test period:  {orders[test_idx[0]].order_date:%Y-%m-%d} to {orders[test_idx[-1]].order_date:%Y-%m-%d}")
print(f"\\nTrain late rate: {y_train_t.mean():.3f}")
print(f"Test late rate:  {y_test_t.mean():.3f}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# --- Comparison Table ---
comparison = pd.DataFrame({
    "Metric": ["F1", "Precision", "Recall"],
    "Random Split": [
        f1_score(y_test_r, y_pred_r),
        precision_score(y_test_r, y_pred_r),
        recall_score(y_test_r, y_pred_r),
    ],
    "Temporal Split": [
        f1_score(y_test_t, y_pred_t),
        precision_score(y_test_t, y_pred_t),
        recall_score(y_test_t, y_pred_t),
    ],
})
comparison["Delta"] = comparison["Temporal Split"] - comparison["Random Split"]
print(comparison.to_string(index=False))
"""
    )
)

cells.append(
    nbformat.v4.new_markdown_cell(
        """### Decision Box: Temporal Validation

> **Finding:** [Fill after running]
>
> **Decision:** Default to temporal split in production training. Random split retained for comparison.
>
> **Impact:** More honest performance estimate.
"""
    )
)

# --- Section 2: Calibration ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 2. Calibration Diagnostic

**Question:** When the model says "70% chance of late delivery," is it actually late 70% of the time?

XGBoost with `binary:logistic` objective often produces well-calibrated probabilities.
Let's verify.
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Calibration on temporal test set (more realistic)
prob_true, prob_pred = calibration_curve(y_test_t, y_proba_t, n_bins=10, strategy="uniform")
brier = brier_score_loss(y_test_t, y_proba_t)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(prob_pred, prob_true, "o-", color="#FF6B6B", label=f"XGBoost (Brier={brier:.4f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve (Reliability Diagram) - Temporal Test Set")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "interim" / "calibration_curve.png", dpi=150)
plt.show()

print(f"Brier Score: {brier:.4f}")
print(f"(Lower is better. Perfect = 0.0, Random = 0.25)")
print(f"\\nCalibration well-calibrated? Brier < 0.15 threshold: {'YES' if brier < 0.15 else 'NO - consider Platt/isotonic'}")
"""
    )
)

cells.append(
    nbformat.v4.new_markdown_cell(
        """### Decision Box: Calibration

> **Finding:** [Fill after running]
>
> **Decision:** If well-calibrated, document finding. If not, apply isotonic regression.
>
> **Impact:** Validates that probability outputs can be used for threshold optimization.
"""
    )
)

# --- Section 3: Threshold Optimization ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 3. Cost-Sensitive Threshold Optimization

**Business context:**
- False Negative (missed late delivery) costs ~3x False Positive (unnecessary alert)
- Cost ratio: C_FN = 3, C_FP = 1
- Optimal threshold minimizes total business cost

Default threshold of 0.5 assumes equal costs - suboptimal for supply chain.
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Sweep thresholds on temporal test set
C_FP = 1.0  # Cost of false alarm (unnecessary intervention)
C_FN = 3.0  # Cost of missed late delivery (reputation, support)

thresholds = np.arange(0.05, 0.95, 0.01)
costs = []
f1_scores_list = []
precisions = []
recalls = []

for t in thresholds:
    y_pred_t_thresh = (y_proba_t >= t).astype(int)
    cm = confusion_matrix(y_test_t, y_pred_t_thresh)
    tn, fp, fn, tp = cm.ravel()

    cost = C_FP * fp + C_FN * fn
    costs.append(cost)
    f1_scores_list.append(f1_score(y_test_t, y_pred_t_thresh, zero_division=0))
    precisions.append(precision_score(y_test_t, y_pred_t_thresh, zero_division=0))
    recalls.append(recall_score(y_test_t, y_pred_t_thresh, zero_division=0))

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Minimum cost: {optimal_cost:.0f}")
print(f"\\nAt optimal threshold:")
print(f"  F1:        {f1_scores_list[optimal_idx]:.4f}")
print(f"  Precision: {precisions[optimal_idx]:.4f}")
print(f"  Recall:    {recalls[optimal_idx]:.4f}")
print(f"\\nAt default 0.5:")
default_idx = np.argmin(np.abs(thresholds - 0.5))
print(f"  F1:        {f1_scores_list[default_idx]:.4f}")
print(f"  Precision: {precisions[default_idx]:.4f}")
print(f"  Recall:    {recalls[default_idx]:.4f}")
print(f"  Cost:      {costs[default_idx]:.0f}")
print(f"\\nCost reduction: {costs[default_idx] - optimal_cost:.0f} ({(costs[default_idx] - optimal_cost) / costs[default_idx] * 100:.1f}%)")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cost curve
ax1.plot(thresholds, costs, color="#FF6B6B", linewidth=2)
ax1.axvline(optimal_threshold, color="green", linestyle="--", label=f"Optimal: {optimal_threshold:.2f}")
ax1.axvline(0.5, color="gray", linestyle=":", label="Default: 0.50")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Total Business Cost (C_FP=1, C_FN=3)")
ax1.set_title("Cost-Sensitive Threshold Optimization")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Precision-Recall tradeoff
ax2.plot(thresholds, precisions, label="Precision", color="blue")
ax2.plot(thresholds, recalls, label="Recall", color="red")
ax2.plot(thresholds, f1_scores_list, label="F1", color="green", linewidth=2)
ax2.axvline(optimal_threshold, color="green", linestyle="--", alpha=0.5)
ax2.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Score")
ax2.set_title("Precision / Recall / F1 vs Threshold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "interim" / "threshold_optimization.png", dpi=150)
plt.show()
"""
    )
)

cells.append(
    nbformat.v4.new_markdown_cell(
        """### Decision Box: Threshold Optimization

> **Finding:** [Fill after running]
>
> **Decision:** Update default threshold from 0.5 to [optimal value].
> Business justification: missed late delivery (FN) costs 3x more than a false alarm (FP).
>
> **Impact:** Recall improves, precision slightly drops. Net business cost decreases.
"""
    )
)

# --- Section 4: Error Analysis ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 4. Error Analysis by Segment

**Question:** Where does the model systematically fail?
Segment prediction errors by shipping mode, region, and order value.
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Use optimal threshold for error analysis
y_pred_optimal = (y_proba_t >= optimal_threshold).astype(int)

# Build analysis dataframe for test set
test_orders = [orders[i] for i in test_idx]
error_df = pd.DataFrame({
    "shipping_mode": [o.shipping_mode for o in test_orders],
    "region": [o.order_region for o in test_orders],
    "benefit": [o.benefit_per_order for o in test_orders],
    "y_true": y_test_t,
    "y_pred": y_pred_optimal,
    "y_proba": y_proba_t,
})
error_df["correct"] = error_df["y_true"] == error_df["y_pred"]
error_df["error_type"] = "correct"
error_df.loc[(error_df["y_true"] == 1) & (error_df["y_pred"] == 0), "error_type"] = "false_negative"
error_df.loc[(error_df["y_true"] == 0) & (error_df["y_pred"] == 1), "error_type"] = "false_positive"

# Add value quartiles
error_df["value_quartile"] = pd.qcut(error_df["benefit"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

print("Overall error rate:", f"{1 - error_df['correct'].mean():.3f}")
print(f"False negatives: {(error_df['error_type'] == 'false_negative').sum()}")
print(f"False positives: {(error_df['error_type'] == 'false_positive').sum()}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Error rate by shipping mode
print("\\n=== Error Rate by Shipping Mode ===")
mode_errors = error_df.groupby("shipping_mode").agg(
    total=("correct", "count"),
    error_rate=("correct", lambda x: 1 - x.mean()),
    fn_rate=("error_type", lambda x: (x == "false_negative").mean()),
    fp_rate=("error_type", lambda x: (x == "false_positive").mean()),
).round(4)
print(mode_errors.to_string())

print("\\n=== Error Rate by Region ===")
region_errors = error_df.groupby("region").agg(
    total=("correct", "count"),
    error_rate=("correct", lambda x: 1 - x.mean()),
    fn_rate=("error_type", lambda x: (x == "false_negative").mean()),
    fp_rate=("error_type", lambda x: (x == "false_positive").mean()),
).round(4)
print(region_errors.to_string())

print("\\n=== Error Rate by Value Quartile ===")
value_errors = error_df.groupby("value_quartile").agg(
    total=("correct", "count"),
    error_rate=("correct", lambda x: 1 - x.mean()),
    fn_rate=("error_type", lambda x: (x == "false_negative").mean()),
    fp_rate=("error_type", lambda x: (x == "false_positive").mean()),
).round(4)
print(value_errors.to_string())
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Visualization - error rates by segment
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# By shipping mode
mode_errors["error_rate"].plot(kind="bar", ax=axes[0], color="#FF6B6B")
axes[0].set_title("Error Rate by Shipping Mode")
axes[0].set_ylabel("Error Rate")
axes[0].axhline(1 - error_df["correct"].mean(), color="gray", linestyle="--", label="Overall")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=45)

# By region
region_errors["error_rate"].plot(kind="bar", ax=axes[1], color="#4ECDC4")
axes[1].set_title("Error Rate by Region")
axes[1].set_ylabel("Error Rate")
axes[1].axhline(1 - error_df["correct"].mean(), color="gray", linestyle="--", label="Overall")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=45)

# By value quartile
value_errors["error_rate"].plot(kind="bar", ax=axes[2], color="#45B7D1")
axes[2].set_title("Error Rate by Order Value Quartile")
axes[2].set_ylabel("Error Rate")
axes[2].axhline(1 - error_df["correct"].mean(), color="gray", linestyle="--", label="Overall")
axes[2].legend()
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "interim" / "error_analysis.png", dpi=150)
plt.show()

# Flag segments with >2x average error
avg_error = 1 - error_df["correct"].mean()
print(f"\\nOverall error rate: {avg_error:.3f}")
print(f"Segments with >2x average error rate ({2 * avg_error:.3f}):")
for mode, row in mode_errors.iterrows():
    if row["error_rate"] > 2 * avg_error:
        print(f"  - Shipping Mode '{mode}': {row['error_rate']:.3f} ({row['error_rate'] / avg_error:.1f}x)")
for region, row in region_errors.iterrows():
    if row["error_rate"] > 2 * avg_error:
        print(f"  - Region '{region}': {row['error_rate']:.3f} ({row['error_rate'] / avg_error:.1f}x)")
for quartile, row in value_errors.iterrows():
    if row["error_rate"] > 2 * avg_error:
        print(f"  - Value '{quartile}': {row['error_rate']:.3f} ({row['error_rate'] / avg_error:.1f}x)")
"""
    )
)

cells.append(
    nbformat.v4.new_markdown_cell(
        """### Decision Box: Error Analysis

> **Finding:** [Fill after running]
>
> **Decision:** If any segment has >2x error rate, add disclaimer to dashboard prediction tab.
>
> **Impact:** Honest about model limitations.
"""
    )
)

# --- Section 5: Feature Engineering V2 ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 5. Feature Engineering V2 - Point-in-Time Customer History

**Hypothesis:** Customer-level features (order count, historical late rate) add signal.

**Critical:** Must be computed point-in-time - only using orders BEFORE the current one.
Naive approach (using all orders) leaks the target.
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Sort orders chronologically
sorted_orders = sorted(orders, key=lambda o: o.order_date)

# Build point-in-time customer features
customer_history: dict[int, list[tuple[int, float]]] = {}  # customer_id -> [(late_risk, benefit)]

pit_features = []  # point-in-time features for each order
pit_labels = []

for order in sorted_orders:
    cid = order.order_customer_id
    history = customer_history.get(cid, [])

    # Compute features from PRIOR orders only
    if len(history) == 0:
        customer_order_count = 0
        customer_historical_late_rate = 0.5  # prior (no info)
        customer_avg_benefit = 0.0
    else:
        customer_order_count = len(history)
        customer_historical_late_rate = sum(h[0] for h in history) / len(history)
        customer_avg_benefit = sum(h[1] for h in history) / len(history)

    # Base features + new customer features
    base = extract_features(order)
    base["customer_order_count"] = customer_order_count
    base["customer_historical_late_rate"] = customer_historical_late_rate
    base["customer_avg_benefit"] = customer_avg_benefit
    pit_features.append(base)
    pit_labels.append(order.late_delivery_risk)

    # Update history AFTER extracting features (point-in-time correctness)
    if cid not in customer_history:
        customer_history[cid] = []
    customer_history[cid].append((order.late_delivery_risk, order.benefit_per_order))

pit_labels = np.array(pit_labels)
print(f"Built {len(pit_features)} point-in-time feature vectors")
print(f"New features: customer_order_count, customer_historical_late_rate, customer_avg_benefit")
print(f"\\nCustomer order count stats:")
counts = [f["customer_order_count"] for f in pit_features]
print(f"  Mean: {np.mean(counts):.1f}, Max: {np.max(counts)}, Median: {np.median(counts):.0f}")
print(f"  % first-time customers (count=0): {sum(1 for c in counts if c == 0) / len(counts):.1%}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Train with point-in-time features (temporal split)
cutoff_pit = int(len(pit_features) * 0.8)
raw_train_pit = pit_features[:cutoff_pit]
raw_test_pit = pit_features[cutoff_pit:]
y_train_pit = pit_labels[:cutoff_pit]
y_test_pit = pit_labels[cutoff_pit:]

encoder_pit = FeatureEncoder()
X_train_pit = encoder_pit.fit_transform(raw_train_pit)
X_test_pit = encoder_pit.transform(raw_test_pit)

model_pit = XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
model_pit.train(X_train_pit.values, y_train_pit)
y_proba_pit = model_pit.predict_proba(X_test_pit.values)
y_pred_pit = (y_proba_pit >= optimal_threshold).astype(int)

f1_pit = f1_score(y_test_pit, y_pred_pit)
f1_baseline = f1_score(y_test_t, y_pred_optimal)

print("=== Feature V2 Comparison (Temporal Split, Optimal Threshold) ===")
print(f"Baseline (12 features): F1 = {f1_baseline:.4f}")
print(f"+ Customer PIT (15 features): F1 = {f1_pit:.4f}")
print(f"Delta: {f1_pit - f1_baseline:+.4f}")
print(f"\\nPromotion threshold: +0.02")
print(f"Promote to production? {'YES' if (f1_pit - f1_baseline) > 0.02 else 'NO - insufficient lift'}")
"""
    )
)

cells.append(
    nbformat.v4.new_code_cell(
        """# Compare naive (leaky) vs point-in-time to quantify leakage
# Naive: use ALL of customer's orders (including future ones)
customer_all_history: dict[int, list[tuple[int, float]]] = {}
for order in sorted_orders:
    cid = order.order_customer_id
    if cid not in customer_all_history:
        customer_all_history[cid] = []
    customer_all_history[cid].append((order.late_delivery_risk, order.benefit_per_order))

naive_features = []
for order in sorted_orders:
    cid = order.order_customer_id
    all_hist = customer_all_history[cid]
    # LEAKY: uses all orders including current and future
    base = extract_features(order)
    base["customer_order_count"] = len(all_hist)
    base["customer_historical_late_rate"] = sum(h[0] for h in all_hist) / len(all_hist)
    base["customer_avg_benefit"] = sum(h[1] for h in all_hist) / len(all_hist)
    naive_features.append(base)

# Same temporal split
raw_train_naive = naive_features[:cutoff_pit]
raw_test_naive = naive_features[cutoff_pit:]

encoder_naive = FeatureEncoder()
X_train_naive = encoder_naive.fit_transform(raw_train_naive)
X_test_naive = encoder_naive.transform(raw_test_naive)

model_naive = XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
model_naive.train(X_train_naive.values, y_train_pit)
y_proba_naive = model_naive.predict_proba(X_test_naive.values)
y_pred_naive = (y_proba_naive >= optimal_threshold).astype(int)

f1_naive = f1_score(y_test_pit, y_pred_naive)

print("=== Leakage Quantification ===")
print(f"Baseline (no customer features): F1 = {f1_baseline:.4f}")
print(f"Point-in-time (correct):         F1 = {f1_pit:.4f}")
print(f"Naive (leaky):                   F1 = {f1_naive:.4f}")
print(f"\\nLeakage inflation: {f1_naive - f1_pit:+.4f} (fake F1 gain from future data)")
print(f"Real signal:       {f1_pit - f1_baseline:+.4f} (actual improvement from customer history)")
"""
    )
)

cells.append(
    nbformat.v4.new_markdown_cell(
        """### Decision Box: Feature Engineering V2

> **Finding:** [Fill after running]
>
> **Decision:** Promote to domain/services.py if F1 lift > 0.02. Otherwise document as explored.
>
> **Impact:** Either improves model OR demonstrates leakage awareness.
"""
    )
)

# --- Section 6: Summary ---
cells.append(
    nbformat.v4.new_markdown_cell(
        """## 6. Summary of Findings & Decisions

| Analysis | Finding | Decision | Impact |
|----------|---------|----------|--------|
| Temporal Split | [F1 gap] | Default to temporal | Honest evaluation |
| Calibration | [Brier score] | [Keep/fix] | Validated probabilities |
| Threshold | [Optimal value] | Update default | Cost reduction |
| Error Analysis | [Worst segments] | [Promote/document] | Model limitations documented |
| Feature V2 | [PIT lift] | [Promote/document] | Leakage quantified |
"""
    )
)

nb.cells = cells
nbformat.write(nb, NOTEBOOK_PATH)
print(f"Notebook written to {NOTEBOOK_PATH}")
