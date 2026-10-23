# Limitations & Decisions

> The honest read on what this project proves, what it doesn't, and when a rule beats the model.
> Every number below traces to a saved artifact: [`reports/model_comparison.json`](../reports/model_comparison.json),
> [`data/metrics/full_run_metrics.json`](../data/metrics/full_run_metrics.json), and [`data/metrics/full_dataset_stats.json`](../data/metrics/full_dataset_stats.json).

---

## TL;DR

A leakage-safe classifier was shipped end-to-end (hexagonal architecture, MLflow tracking, SHAP, 156 tests). The model works — but it does **not** beat a simple rule by enough to justify its operational complexity. **That is the finding, and it is a valid business conclusion**, not a failed project. The recommended production design is a shipping-mode rule plus a model only for the genuinely ambiguous order tier.

---

## 1. The dataset is tutorial-grade

- **DataCo Smart Supply Chain**: 180,519 order line items across 65,752 orders.
- **54.82% late delivery rate** (36,048 late orders). A real logistics network with a coin-flip late rate would be in crisis; this points to dataset semantics, not a real operation.
- There is no live feed, no concept drift, and no cost data. All business-impact figures in [`BUSINESS_IMPACT.md`](./BUSINESS_IMPACT.md) are therefore **illustrative and clearly labeled as such**.

**Decision:** Treat this as a *methodology* showcase (leakage control, validation, explainability, production scaffolding), not a claim of operational savings at a named company.

---

## 2. One feature does almost all the work

After removing the three leakage columns, late delivery is **near-deterministic from shipping mode**:

| Shipping mode | Late rate |
|---------------|-----------|
| First Class   | **95.3%** |
| Second Class  | **76.7%** |
| Same Day      | 46.2%     |
| Standard Class| 38.1%     |

SHAP confirms the model learned exactly this — the top features for XGBoost are the shipping-mode dummies (mean \|SHAP\| 0.616 / 0.432 / 0.134), with every other feature an order of magnitude smaller.

**Implication:** If you know the shipping mode, you already know most of the answer. A model that ingests 12 features is largely re-deriving a 4-row lookup table.

---

## 3. The complex model adds no meaningful lift

| Model | Split | F1 | Precision | Recall | AUC-ROC |
|-------|-------|----|-----------|--------|---------|
| Logistic Regression (baseline) | temporal | 0.6531 | 0.8515 | 0.5297 | 0.7241 |
| XGBoost (depth 6) | temporal | **0.6648** | 0.7743 | 0.5825 | 0.7233 |
| XGBoost (depth 6) | random | 0.6543 | 0.8448 | 0.5338 | 0.7233 |

- XGBoost beats logistic regression by **~0.012 F1** on the temporal split — within noise for an ops decision.
- AUC-ROC is effectively tied (0.7233 vs 0.7241).
- **Customer-history features added zero lift** — shipping mode dominates so completely that engineered behavioural features did not move the metric.

**Decision:** Do not ship XGBoost as the headline model on the strength of a 0.012 F1 gap. The gap does not survive the cost of maintaining a gradient-boosted model, its calibration, and its retraining cadence versus a transparent rule.

---

## 4. When the rule beats the model — and when it doesn't

| Order tier | Late rate | Best tool | Why |
|------------|-----------|-----------|-----|
| First Class | 95.3% | **Rule: always flag** | The model can't beat "always say late" here. |
| Second Class | 76.7% | **Rule: always flag** | Same — high enough that a rule captures it. |
| Same Day | 46.2% | Rule: flag, monitor | Near base rate; low volume; cheap to just flag. |
| Standard Class | 38.1% | **Model adds value here** | The only tier where the answer is genuinely uncertain and features beyond mode can discriminate. |

**Recommended production design:** a deterministic flag on First/Second Class, plus the model scoring *only* the Standard Class tier where it has room to discriminate. This is simpler, auditable, and concentrates the ML where it earns its keep.

---

## 5. Calibration & operating point

- Out-of-the-box XGBoost is **poorly calibrated** (Brier = 0.2028). An isotonic `CalibratedModelAdapter` was added and unit-tested, but it is **not wired** into `train.py` or the Streamlit serving path — live scores are uncalibrated probabilities.
- A pure expected-cost sweep under FN ≈ 3× FP picks threshold **0.05** (recall 1.0, flag rate 1.0) — flag everything. That minimizes the toy cost function but is operationally useless.
- The demo / default serving threshold is **0.35** (ops compromise): measured recall **0.8013**, precision **0.6145**, flag rate ≈ **0.72** on the temporal test set ([`reports/model_comparison.json`](../reports/model_comparison.json)).

**Decision:** Document the cost asymmetry and the ops compromise explicitly. Do not call 0.35 "cost-optimal" — that label belongs to 0.05 under the toy cost model.

---

## 6. Validation honesty

- **Temporal split is the headline** (train on past orders, test on future): F1 0.6648 vs random-split 0.6543 — no degradation, so the model is temporally stable.
- **Split-before-encode** is enforced (encoder fit on training data only) to prevent preprocessing leakage.
- **Three leakage columns** (`Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`) are dropped at the data adapter and guarded by a test.
- No segment-specific error bias was found across the K-Means risk cohorts.

---

## 7. What this project still proves

Even though the model does not beat the rule, the work still shows:

- **Leakage control** as a first-class concern, enforced in code and tests.
- **Leakage-safe, temporally-validated** evaluation with the right metric (F1, not accuracy).
- **Calibration awareness** (Brier documented; adapter available) and an explicitly chosen ops threshold (0.35), not a silent 0.5 default.
- **Explainability** (global + local SHAP) used to *interrogate* the model, which is how the shipping-mode dominance was caught.
- **Production scaffolding**: hexagonal architecture, MLflow tracking + registry, 230 tests at 95% coverage, CI, a live Streamlit dashboard.
- **The judgement to recommend the simpler tool** when the data says so.

---

*See also: [`BUSINESS_IMPACT.md`](./BUSINESS_IMPACT.md) · [`reports/model_comparison.json`](../reports/model_comparison.json) · [`ARCHITECTURE.md`](./ARCHITECTURE.md)*
