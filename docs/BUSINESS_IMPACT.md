# Business Impact (Illustrative)

> **Read this first.** The DataCo dataset ships **no cost data** and is tutorial-grade (54.82% late rate).
> Every dollar figure below is **illustrative** — built from a transparent formula with **labeled assumptions** you
> can change. The *model performance* numbers are real and traceable
> ([`reports/model_comparison.json`](../reports/model_comparison.json)); the *cost* numbers are scenario inputs, not measured outcomes.
> **Do not put any dollar figure on a résumé unless it is computed here with stated assumptions.**

---

## 1. Assumptions (edit these)

Operating point used below: **threshold 0.35** (the demo / ops compromise — see `LIMITATIONS_AND_DECISIONS.md`).
Default 0.5 metrics are also frozen in `model_comparison.json` if you want the stricter flag set.

| Symbol | Meaning | Value used | Source / status |
|--------|---------|-----------|-----------------|
| `N` | Orders evaluated per period | 10,000 | Per-10k basis for readability (dataset has 65,752 orders) |
| `late_rate` | Base late-delivery rate | **54.82%** | Measured — `data/metrics/full_dataset_stats.json` |
| `R` | Model recall @ threshold 0.35 | **0.8013** | Measured — XGBoost temporal, `model_comparison.json` |
| `P` | Model precision @ threshold 0.35 | **0.6145** | Measured — XGBoost temporal, `model_comparison.json` |
| `C_late` | Cost of an unmitigated late delivery (service + refund/reship + churn) | **$15** | **Illustrative** — industry range $5–$25; midpoint chosen |
| `r` | Fraction of `C_late` recovered when a late order is caught early | **40%** | **Illustrative assumption** — proactive reroute/notify rarely recovers 100% |
| `C_int` | Cost of one proactive intervention (ops time, expedite) | **$3** | **Illustrative assumption** |

> Change any value and the conclusion may flip. The point of this doc is the **formula and the break-even**, not a headline number.

---

## 2. Order economics at threshold 0.35

Per **N = 10,000** orders, using the measured rates:

| Quantity | Formula | Value |
|----------|---------|-------|
| Actual late orders | `N × late_rate` | 5,482 |
| Late orders **caught** (true positives) | `late × R` | 4,393 |
| Late orders **missed** (false negatives) | `late − caught` | 1,089 |
| Orders **flagged** for intervention | `caught / P` | 7,149 |
| **False alarms** (false positives) | `flagged − caught` | 2,756 |

So the model asks ops to act on ~7,149 of every 10,000 orders, correctly catches ~80% of late orders, and ~39% of its flags are false alarms. That is the cost of higher recall.

---

## 3. Net-value formula

```
Net value = (caught × C_late × r)  −  (flagged × C_int)
          = value recovered on caught late orders  −  cost of acting on all flags
```

**Worked example (assumptions above):**

```
Value recovered = 4,393 × $15 × 0.40   = $26,358
Intervention cost = 7,149 × $3          = $21,447
Net value (per 10,000 orders)           ≈ $4,911
```

Roughly **$0.49 of net value per order screened** — *under these illustrative assumptions*. The figure is positive but modest, which is consistent with the project's core finding: the signal is real but shallow, and a high-recall threshold is expensive in ops load.

---

## 4. Sensitivity — the answer depends entirely on `C_late` and `r`

Net value per 10,000 orders (`C_int` = $3 fixed):

| `C_late` ↓ / `r` → | r = 20% | r = 40% | r = 60% |
|--------------------|--------:|--------:|--------:|
| **$5**  | −$12,654 | −$8,257 | −$3,860 |
| **$15** | −$8,257 | **+$4,911** | +$18,079 |
| **$25** | −$3,860 | +$18,079 | +$40,018 |

**Break-even** (net value = 0) is `caught × C_late × r = flagged × C_int`, i.e. `C_late × r = 21,447 / 4,393 ≈ $4.88`. Below that the program loses money; above it, it pays. This break-even is the number to defend, not a fixed ROI.

---

## 5. Rule-based alternative (the recommended design)

The model is **not** the cheapest way to capture most of the value. Late rate is near-deterministic by shipping mode (`full_dataset_stats.json`):

| Mode | Late rate | Rule | Effective precision of "flag all" |
|------|-----------|------|-----------------------------------|
| First Class | 95.3% | Always flag | 0.953 |
| Second Class | 76.7% | Always flag | 0.767 |
| Standard Class | 38.1% | **Model scores this tier** | model adds value here |
| Same Day | 46.2% | Flag, monitor | 0.462 |

A **flag-all rule on First + Second Class** captures their late orders at higher precision (0.767–0.953) than the model's blended precision at 0.35 (0.6145) — with zero model maintenance, full auditability, and no calibration risk. The model then earns its keep only on the **Standard Class** tier, where lateness is genuinely uncertain.

> Exact rule-vs-model *recall* depends on the volume share of each mode (visible in the dashboard's full-dataset view); the late-rate-by-mode figures above are measured.

---

## 6. Bottom line

- The model produces **modest, assumption-dependent** net value; it is not a slam-dunk ROI story, and the project does not pretend otherwise.
- The **defensible recommendation** is a shipping-mode rule for the high-risk tiers plus a model on the ambiguous Standard Class tier — cheaper, transparent, and nearly as effective.
- The value of this analysis is the **decision discipline**: measuring lift honestly, pricing the cost asymmetry (ops threshold 0.35 vs pure cost-opt 0.05), and recommending the simpler tool when the data says so.

*See [`LIMITATIONS_AND_DECISIONS.md`](./LIMITATIONS_AND_DECISIONS.md) for the modelling rationale and [`reports/model_comparison.json`](../reports/model_comparison.json) for frozen metrics.*
