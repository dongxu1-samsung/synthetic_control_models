# Synthetic Control Measurement Pipeline -- Flight 648768 (Hulu)

Campaign 291260, Flight 648768 (Hulu app, App ID 3201601007625).

## Pipeline

Run in order from the `measurement/` directory:

1. `scripts/01_load_and_clean_hulu.py` -- loads the CSV export of
   `UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.PROPENSITY_FEATURES_HULU`
   (built by `feature_query_hulu.sql`), cleans types, saves to Parquet.
2. `scripts/02_fit_propensity_hulu.py` -- fits a logistic regression exposure-
   propensity model on pre-period covariates only (excludes
   `CAMPAIGN_IMPRESSIONS` and `CONVERTED_POST_EXPOSURE`, which are
   post-treatment/outcome). Scores every unit.
3. `scripts/03_match_and_weight_hulu.py` -- common-support trimming, then two
   bias-correction approaches: 1:1 nearest-neighbor caliper matching, and
   stabilized/trimmed IPTW.
4. `scripts/03b_derandomized_matching_hulu.py` -- de-randomized (jitter-averaged)
   re-run of the caliper matching, as a precautionary check for the tie-breaking
   bias found on the Apple TV pipeline (see below -- not actually present here).
5. `scripts/04_balance_diagnostics_hulu.py` -- Standardized Mean Difference (SMD)
   table: raw vs. matched vs. IPTW, for all 33 pre-period covariates.
6. `scripts/05_lift_estimation_hulu.py` -- lift estimates (naive / matched / IPTW)
   with confidence intervals and significance tests.
7. `scripts/06_robustness_checks_hulu.py` -- placebo test (split control pool into
   two pseudo-groups, both truly unexposed; measured lift should be ~0) plus
   matching-reuse diagnostics.
8. `scripts/06c_placebo_iptw_hulu.py` -- IPTW placebo test, refitting the
   propensity model on the pseudo-split (avoids the reuse-of-real-score bug
   found on the Apple TV pipeline).
9. `scripts/07_final_summary_hulu.py` -- consolidated final lift table.

## Headline result

| Method | Conv. rate (test) | Conv. rate (control) | Abs. lift | Rel. lift | Placebo-validated |
|---|---|---|---|---|---|
| Naive (unadjusted) | 6.67% | 3.22% | +3.46pp | +107.4% | N/A -- reference only, known biased |
| 1:1 caliper matching (de-randomized) | 6.67% | 3.48% | +3.19pp | **+91.5%** | Yes (p=0.84 single-run, p=0.47 de-randomized) |
| IPTW (stabilized, trimmed) | 6.57% | 3.25% | +3.33pp | **+102.3%** | Yes (p=0.76, refit-on-pseudo-split) |

**Both adjusted methods are close to each other and both cleanly placebo-validated**
(unlike the Apple TV pipeline, which showed a wide 9.2% vs. 21.6% divergence
requiring further investigation). For Hulu flight 648768, matching (+91.5%) and
IPTW (+102.3%) agree within ~11pp of relative lift, and every one of the three
placebo checks (naive, matched, IPTW-refit) passed with p > 0.4. The naive/
unadjusted +107.4% is included only to show the scale of RTB selection bias
that adjustment corrects for -- it should not be reported as the actual
campaign lift.

**Recommended headline number: ~+92-102% relative lift** (report as a range
given the two adjusted methods' spread; IPTW's +102.3% is the primary estimate
since it uses the full sample rather than a caliper-restricted subset, but the
methods agree closely enough here that either is defensible).

## Data quality issues found and fixed during this analysis

### 1. Perfect collinearity between HIST_IMP_EVENTS_7D and HIST_TILE_IMPRESSIONS_7D

For this Hulu population, every pre-period ad impression happens to be
tile-channel (`HIST_IMP_EVENTS_7D` and `HIST_TILE_IMPRESSIONS_7D` are
identical, corr = 1.0000, across all 5.96M rows) -- unlike the Apple TV
population, where the collinearity issue was a *zero-variance* column
(`HIST_CTV_IMPRESSIONS_7D`, constant 0). Perfect collinearity makes the
design matrix singular, surfacing as the same
`RuntimeWarning: divide by zero / overflow / invalid value in matmul`
during logistic regression fitting seen on the Apple TV pipeline, but from a
different root cause.

**Fix**: drop `HIST_TILE_IMPRESSIONS_7D` from the propensity model's feature
set (it's fully redundant with `HIST_IMP_EVENTS_7D` for this population);
`CHANNEL_TILE_RATIO` (a derived feature) still carries channel-mix signal.
The dropped column was still checked for balance in the diagnostics step
(balance doesn't require the propensity model to use every raw covariate).

### 2. Same known fixes carried over from the Apple TV pipeline (checked, applied where relevant)

- PSID kept as a string throughout (never cast to int64/float64) --
  verified unique (5,964,077 distinct PSIDs / 5,964,077 rows) immediately
  after loading.
- Zero-variance columns dropped before scaling (confirmed
  `HIST_CTV_IMPRESSIONS_7D` is again constant 0 for this population too).
- Heavy-tailed count/duration covariates `log1p`-transformed before scaling.
- Rare `DEVICE_COUNTRY` categories (15 countries, <1000 rows) bucketed into
  `OTHER` to avoid quasi-separation.
- De-randomized (jitter-averaged) matching run as a precaution against the
  tie-breaking bias found on the Apple TV pipeline -- **result: not actually
  present here**. A single matching run already passed its placebo test
  cleanly (p=0.84); the 10x jittered re-run confirmed the same lift
  (0.03192 vs. 0.03190 single-run) with very low variance (std 0.00019).
- IPTW placebo test refit the propensity model on the pseudo-split (rather
  than reusing the real propensity score, the bug found on the Apple TV
  pipeline's first placebo attempt) -- passed cleanly (AUC 0.5013, p=0.76).

Net result: this Hulu run required **one new fix** (the collinearity drop)
and needed **none** of the Apple TV pipeline's tie-breaking/placebo-script
fixes to be actually invoked (though the precautionary checks were run and
confirmed clean).

## Propensity model fit

Train/test AUC ~0.622 (vs. ~0.72 for Apple TV) -- meaningfully above chance
but a weaker discriminator between exposed/control for this campaign, meaning
covariates explain less of who won the RTB auction here. Top standardized
coefficients: `HIST_IMP_EVENTS_7D` (+1.02), `AVG_DAILY_IMPRESSIONS` (-0.69),
`HIST_IMPRESSIONS_7D` (-0.49), `HIST_ACTIVE_DAYS_7D` (-0.44),
`TOTAL_ACTIVE_DAYS` (-0.35).

## Balance diagnostics

Raw (unadjusted) data: 6/33 covariates with |SMD| > 0.1 (max 0.2314,
`LIGHT_APP_COUNT`). Both matching (max |SMD| 0.0356) and IPTW (max |SMD|
0.0520) bring all 33 covariates under the 0.1 threshold.

## Feature table validation

`UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.PROPENSITY_FEATURES_HULU`: 5,964,077
rows / distinct PSIDs. 734,974 exposed (test), 5,229,103 control (eligible,
unexposed), 217,283 converted -- confirmed to match the source pool
(`SYNTHETIC_CONTROL_MEASUREMENT_USER_POOL_3201601007625`) restricted to
`ELIGIBLE_BID_REQUESTS_ENTERED_AUCTION > 0` exactly.

## Known limitations

- **Weaker propensity model** (AUC 0.622 vs. 0.72 for Apple TV) means more
  residual confounding is plausible if there are unmeasured drivers of RTB
  auction wins for this campaign; the clean balance diagnostics and passing
  placebo tests are reassuring but don't rule this out.
- Feature set still limited to ~40 of the ~600 raw columns available in the
  underlying UDW tables (same limitation noted on the Apple TV pipeline).
- Attribution window for conversions not independently re-verified for this
  flight/conversion group.
