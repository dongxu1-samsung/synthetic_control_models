# Advanced Causal Inference Methods -- Hulu Flight 648768

Follow-up to `README_hulu.md` (matching / IPTW pipeline). Adds:
1. **AIPW (doubly-robust) estimation** with LightGBM nuisance models, cross-fitted
2. **CausalForestDML** for heterogeneous treatment effects (CATE)
3. **CATE-by-segment breakdown** (which users respond most to ad exposure)
4. **Qini/AUUC curve** validating the CATE model's targeting quality

## Pipeline

Run in order from the `measurement/` directory (after the base pipeline in `README_hulu.md`):

8. `scripts/08_aipw_gbm_hulu.py` -- 5-fold cross-fitted LightGBM propensity + outcome
   (T-learner) nuisance models, combined into an AIPW/doubly-robust ATE estimate
   with analytic + bootstrap 95% CIs.
9. `scripts/08b_placebo_aipw_hulu.py` -- placebo test for the AIPW estimator
   (control pool split into two pseudo-groups, nuisance models refit on the
   pseudo-treatment label).
10. `scripts/09_causal_forest_hulu.py` -- EconML `CausalForestDML` (honest forest +
    Double ML orthogonalization) fit on a stratified 750K-row subsample, producing
    a personalized `tau(X)` for every unit plus 95% CI bounds.
11. `scripts/09b_cate_by_segment_hulu.py` -- aggregates CATE by device country,
    app-engagement tier, historical ad-exposure recency, historical impression
    volume, and TV age (STV_YEAR) to surface targeting-relevant heterogeneity.
12. `scripts/10_qini_auuc_hulu.py` -- IPW-adjusted Qini/AUUC curve: ranks users by
    predicted CATE, measures how much of the total campaign lift would be
    captured by targeting only the top-k% (the standard no-ground-truth
    validation for CATE models on real production data).
13. `scripts/11_final_summary_v2_hulu.py` -- consolidated 5-method comparison table.

## Why these methods (vs. matching/IPTW alone)

- **AIPW/doubly-robust**: consistent if *either* the propensity model or the
  outcome model is correctly specified -- strictly more robust than matching
  or IPTW, which both depend entirely on one propensity model. Also lets us
  swap in a more flexible nuisance model (gradient boosting) instead of plain
  logistic regression.
- **CausalForestDML**: every method up to this point reports a single ATE
  number for the whole campaign. A causal forest estimates a *personalized*
  treatment effect for every user, which is directly actionable for
  targeting/budget-allocation decisions in a way ATE alone can't be.
- **Qini/AUUC**: with no ground-truth ITE (real production data, not a
  benchmark dataset), this is the standard way to validate whether a CATE
  model's *ranking* of users by predicted benefit is actually informative --
  IPW-adjustment keeps the evaluation valid despite non-random RTB exposure.

## Results

### AIPW / doubly-robust (LightGBM nuisance models, cross-fitted, full 5.96M population)

- Propensity AUC improved from **0.622** (logistic regression) to **0.668**
  (LightGBM, 5-fold cross-fitted) -- gradient boosting recovers meaningfully
  more separating signal from the same covariates.
- **ATE (AIPW): 3.156pp absolute, +98.1% relative lift**
- 95% CI (bootstrap, 2000 resamples): **[3.09pp, 3.22pp]**
- Placebo test: **passed cleanly** (pseudo-propensity AUC 0.4995, placebo ATE
  0.00006, p=0.73) -- refitting nuisance models on a random pseudo-split of
  the control pool recovers ~0 lift, as expected.

### CausalForestDML (heterogeneous treatment effects)

- Fit on a stratified 750K-row subsample (preserves the 12.3% treatment rate)
  using EconML's honest random forest + Double ML orthogonalization
  (LightGBM as the `model_y`/`model_t` nuisance learners).
- **Mean CATE: 3.114pp absolute, ~96.7% relative lift** -- lands almost
  exactly between matching (91.5%) and IPTW (102.3%), and very close to AIPW
  (98.1%). Four independent methods now converge to a tight band.
- Meaningful heterogeneity found: CATE std 0.0124, range from -0.87pp to
  +14.2pp across users.

### CATE by segment (targeting insights)

| Segment dimension | Highest-CATE group | Lowest-CATE group |
|---|---|---|
| App engagement tier | **No_App_Activity** (CATE 4.88pp) | Light (CATE 2.95pp) |
| Historical ad-exposure recency | **0 prior active days** (CATE 3.77pp) | 2-3 active days (CATE 2.72pp) |
| Device country | UNKNOWN (CATE 3.60pp) | US (CATE 3.02pp) |
| Historical impression volume | Q4 highest (CATE 3.52pp) | Q2 (CATE 2.71pp) |
| TV age (STV_YEAR) | 2020-2021 (CATE 3.32pp) | 2024-2026 (CATE 2.74pp) |

**Key takeaway**: the campaign is most incremental for users with **little or
no prior app engagement / ad-exposure history** -- consistent with the
intuitive story that already-engaged users are more likely to convert
regardless of this specific exposure (lower marginal lift), while previously
low-engagement users see the largest *incremental* effect from being reached.
This is the standard "diminishing returns on already-warm audiences" pattern
seen in uplift modeling, and is directly actionable: budget could be
prioritized toward users flagged with `HIST_ACTIVE_DAYS_7D = 0` or
`ENGAGEMENT_TIER = No_App_Activity`.

Full table: `results_hulu/cate_by_segment_hulu.csv`.

### Qini / AUUC validation

- Qini coefficient (model AUUC minus random-targeting baseline): **0.0307**
  -- a clear, non-trivial improvement over random targeting.
- Targeting the **top 20% of users by predicted CATE captures ~1.8x the
  cumulative gain** that random targeting at the same spend level would
  achieve (decile table, `results_hulu/qini_auuc_hulu.csv`).
- Qini curve (`results_hulu/qini_curve_hulu.png`) shows the classic
  well-behaved shape: steep early rise, peak around 20-25% targeted, gradual
  decline back to the full-population ATE at 100% -- confirms the causal
  forest's CATE ranking carries real, exploitable signal (not noise).
- IPW weights were trimmed at the 1st/99th percentile (same convention as the
  base IPTW step) to avoid extreme-weight instability in small top-k buckets.

## Updated headline (5-method comparison)

| Method | Abs. lift | Rel. lift | Placebo-validated |
|---|---|---|---|
| Naive (unadjusted) | +3.46pp | +107.4% | N/A -- reference only |
| 1:1 caliper matching | +3.19pp | +91.5% | Yes |
| IPTW (stabilized) | +3.33pp | +102.3% | Yes |
| **AIPW (doubly-robust, LightGBM)** | **+3.16pp** | **+98.1%** | Yes |
| **CausalForestDML (mean CATE)** | **+3.11pp** | **+96.7%** | N/A (consistent w/ others) |

**All four adjusted methods now converge within a ~10.9pp relative-lift band
(91.5%-102.3%)**, with the two most methodologically robust estimators (AIPW
and the causal forest) landing almost exactly in the middle. Recommended
headline: **~92-102% relative conversion lift**, with the causal forest's
segment breakdown as an actionable targeting overlay on top of the ATE
number.

## What we did NOT try, and why

Per the accompanying discussion: SOTA deep-learning causal models (MOCA,
DDRNet, TransDCA, diffusion-based IWDD, etc.) were considered but not
implemented here. In the author's own 17-model/15-dataset causal inference
benchmark, these architectures show their advantage on high-dimensional
(1000+ feature) or unstructured (text/image-derived) covariate spaces, or
when continuous-treatment dose-response curves are needed. Hulu's setup
(~40 tabular features, binary treatment, binary outcome, 6M rows) is the
opposite of that regime -- tree/DML-based methods (CausalForestDML,
LightGBM-nuisance AIPW) match or beat deep nets on exactly this kind of
problem in that benchmark, train in minutes instead of hours, and are what's
actually deployed in production ad-tech. Revisit only if a fourth/fifth
opinion is needed to adjudicate a future disagreement between AIPW/DML-style
estimators (as happened on the Apple TV pipeline's unresolved matching-vs-IPTW
divergence).
