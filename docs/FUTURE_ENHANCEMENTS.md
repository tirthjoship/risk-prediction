# Future Enhancements — Optional Depth

**Status:** Optional follow-ons beyond what is already shipped and documented.
**Not a gap list.** Calibration measurement (Brier 0.2028), an ops threshold (0.35), and the rule-vs-model recommendation are already in the repo. Items below are *extensions*, not unfinished core work.

---

## Scope boundary

This project owns **operations / logistics late-delivery risk** (predictive ML + operating-point decisions + MLOps).
Causal / uplift / product-experimentation work belongs in a separate project — do not duplicate either way.

---

## Optional extensions

| ID | Idea | Why it is optional | Notes |
|----|------|--------------------|-------|
| E1 | Publish a full cost-threshold artifact (`reports/cost_threshold_analysis.*`) | Sweep already run in the deep-dive notebook; freeze as a standalone report | Ops point 0.35 and pure cost-opt 0.05 are already in `model_comparison.json` |
| E2 | Wire isotonic calibration into train + serving | Adapter exists and is tested; pipeline intentionally left uncalibrated for now | Documented in `ARCHITECTURE.md` / `LIMITATIONS` |
| E3 | Drift monitoring design (PSI/KS + retraining policy) | MLOps credential; not required for the decision narrative | Design doc + small `monitoring/` module if pursued |
| E4 | Prescriptive carrier / mode assignment under a late-risk budget | Would add a true optimization layer on top of the predictor | Explicitly *out of scope* for the current project scope |

## Explicitly rejected / deferred

| Item | Why |
|---|---|
| Uplift modeling / causal what-ifs | Out of scope |
| Demand forecasting | DataCo too weak for credible time series; not this project's identity |
| Deep models / AutoML sweep | Shipping-mode dominance means the ceiling is data-limited |

---

*Shipped decision narrative: [`LIMITATIONS_AND_DECISIONS.md`](./LIMITATIONS_AND_DECISIONS.md) · [`BUSINESS_IMPACT.md`](./BUSINESS_IMPACT.md) · [`reports/model_comparison.json`](../reports/model_comparison.json)*
