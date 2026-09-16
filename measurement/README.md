# Synthetic Control Measurement Pipeline -- Flight 658095 (Apple TV)

Campaign 296636, Flight 658095, Conversion Group 5534 (Apple TV app, App ID 3201807016597).

## Pipeline

Run in order from the `measurement/` directory:

1. `scripts/01_load_and_clean.py` -- loads the CSV export of
   `UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.PROPENSITY_FEATURES_APPLE`
   (built by `feature_query.sql`), cleans types, saves to Parquet.
2. `scripts/02_fit_propensity.py` -- fits a logistic regression exposure-
   propensity model on pre-period covariates only (excludes
   `CAMPAIGN_IMPRESSIONS` and `CONVERTED_POST_EXPOSURE`, which are
   post-treatment/outcome). Scores every unit.
3. `scripts/03_match_and_weight.py` -- common-support trimming, then two
   bias-correction approaches: 1:1 nearest-neighbor caliper matching, and
   stabilized/trimmed IPTW.
4. `scripts/03b_derandomized_matching.py` -- de-randomized (jitter-averaged)
   re-run of the caliper matching to correct a tie-breaking bias found
   during robustness checks (see below).
5. `scripts/04_balance_diagnostics.py` -- Standardized Mean Difference (SMD)
   table: raw vs. matched vs. IPTW, for all 33 pre-period covariates.
6. `scripts/05_lift_estimation.py` -- lift estimates (naive / matched / IPTW)
   with confidence intervals and significance tests.
7. `scripts/06_robustness_checks.py` -- placebo test (split control pool into
   two pseudo-groups, both truly unexposed; measured lift should be ~0) plus
   matching-reuse diagnostics.
8. `scripts/06b_placebo_iptw.py` / `06c_placebo_iptw_fixed.py` -- IPTW placebo
   test; `06b` has a known bug (see below), `06c` is the corrected version.
9. `scripts/07_final_summary.py` -- consolidated final lift table.
10. `scripts/08_population_scope_check.py` -- diagnostic: checks whether the
    matched-vs-IPTW gap is due to population scope (it is not -- see below).
11. `scripts/09_ps_stratified_lift.py` -- diagnostic: lift broken down by
    propensity-score decile; found genuine effect heterogeneity that
    explains the matched-vs-IPTW gap.
12. `scripts/10_aipw_estimator.py` -- doubly-robust AIPW estimator, used as
    the reconciled headline estimate (see below).

## Headline result

| Method | Conv. rate (test) | Conv. rate (control) | Abs. lift | Rel. lift | Placebo-validated |
|---|---|---|---|---|---|
| Naive (unadjusted) | 10.38% | 7.39% | +2.99pp | +40.4% | N/A -- reference only, known biased |
| 1:1 caliper matching (de-randomized) | 10.38% | 9.51% | +0.87pp | +9.2% | Yes (after jitter fix) |
| **AIPW (doubly robust)** | -- | -- | **+1.24pp** | **+15.0%** | N/A (cross-fitted, not placebo-tested directly) |
| IPTW (stabilized, trimmed) | 9.71% | 7.99% | +1.73pp | +21.6% | Yes (after placebo-script fix) |

**Recommended headline number: AIPW, +15.0% relative lift (95% CI [+11.2%,
+18.8%], absolute [0.92pp, 1.55pp]).** AIPW is doubly robust (consistent if
either the propensity or outcome model is correctly specified) and combines
information from both, rather than relying on one weighting scheme -- see
"Divergence resolved" below for why matching and IPTW disagree in the first
place (genuine effect heterogeneity, not bias) and why AIPW is the right
tie-breaker. The naive/unadjusted 40.4% is included only to show the scale
of RTB selection bias that adjustment corrects for -- it should not be
reported as the actual campaign lift.

**The heterogeneity finding is arguably more actionable than any single
number**: incremental lift ranges from +90% among users unlikely to be
reached organically down to -3% among users who were likely to convert
regardless of exposure (see `results/lift_by_ps_decile.csv`) -- this has
direct implications for targeting strategy.


## Data quality issues found and fixed during this analysis

### 1. PSID precision loss during CSV load (critical, fixed)

Snowflake stores `PSID` as `NUMBER(38,0)`. Several real PSID values exceed
the signed 64-bit integer range (e.g. `15193463195813936410` >
`9223372036854775807`). The original loader used
`pd.to_numeric(errors="coerce")`, which round-trips through `float64`
(53-bit mantissa) -- this silently collapsed millions of distinct large
PSIDs onto the same rounded value, with huge numbers landing on the exact
`INT64_MIN`/`INT64_MAX` sentinels. Confirmed empirically: total rows dropped
from 5,742,285 (correct, matches Snowflake `COUNT(DISTINCT PSID)`) to
856,035 distinct values after the buggy cast, with one sentinel value
alone absorbing 1.95M rows (34% of the dataset).

**Fix**: `PSID` is now loaded and kept as a `string`, never cast to
`int64`/`float64` (it is only ever used as a join/grouping key, never in
arithmetic). `01_load_and_clean.py` now asserts
`df["PSID"].nunique() == len(df)` immediately after loading and will raise
if this invariant is violated.

**Impact on results**: the propensity model itself was unaffected (PSID is
never used as a model feature). Matching/lift point estimates were also
unaffected, because nearest-neighbor selection operates on
`PROPENSITY_SCORE`, not `PSID`. The bug only corrupted *diagnostics* that
count/track PSID identity (e.g. "how many unique controls were reused"),
which looked alarmingly bad pre-fix (7.4x avg reuse, one control ID
"reused" 534,634 times) and are now much healthier (1.52x avg reuse, median
reuse = 1x) post-fix.

### 2. Zero-variance and extreme-cardinality features breaking StandardScaler (fixed)

`HIST_CTV_IMPRESSIONS_7D` is constant 0 for every row in this flight's
population (no CTV-tagged impressions in the pre-period). `StandardScaler`
divides by the column's standard deviation, so a constant column produces
0/0 during scaling -- this surfaced as `RuntimeWarning: divide by zero /
overflow / invalid value in matmul` during logistic regression fitting.
**Fix**: drop zero-variance columns before scaling.

### 3. Heavy-tailed count/duration features causing numerical overflow (fixed)

Several raw count/duration covariates (e.g. `HIST_CLICK_EVENTS_7D`,
`TOTAL_VOD_TIME_7D`) have a small number of power-user rows with values
100-500+ standard deviations above the mean after z-scoring. Feeding these
directly into gradient-based logistic regression caused numerical overflow
during optimization. **Fix**: `log1p`-transform heavy-tailed count/duration
columns before scaling (standard practice for this feature type).

### 4. Rare categorical levels causing quasi-separation (fixed)

`DEVICE_COUNTRY` had several singleton/near-singleton categories (e.g. `AE`,
`AR` with 1 row each) which can perfectly separate `EXPOSED` for that single
row, driving the corresponding one-hot coefficient toward +-infinity during
optimization. **Fix**: bucket categories with <1,000 rows into `OTHER`.

### 5. 1:1 matching tie-breaking bias (found via placebo test, fixed)

Placebo test (split the eligible-control pool into two random,
both-truly-unexposed pseudo-groups; measured lift should be ~0): the
original nearest-neighbor matcher **failed** this test (measured lift
+0.19pp, t=7.46, p<0.0001) even though the naive/unadjusted comparison on
the same pseudo-split passed cleanly (lift ~0.05pp, p=0.06). Root cause:
heavy ties in the propensity score (driven by discrete/categorical
covariates -- one score value alone is shared by ~50,000 control rows)
make `sklearn.NearestNeighbors`' tie-breaking deterministic by array
position, which can introduce a small systematic bias at this sample size.

**Fix**: add small random jitter (1e-6, far below the 0.01 caliper) to the
propensity score before matching, repeat 10x with independent jitter draws,
and average. Re-running the placebo test with de-randomized matching passes
cleanly (mean placebo lift 0.003pp, t=0.36, p=0.73). The de-randomized real
lift estimate (8.70pp mean, range 8.35-9.00pp across 10 repeats) is close to
the original single-run estimate (8.75pp) -- the bug had negligible
*practical* impact on the point estimate, but was worth finding and fixing
rigorously since it could matter more in other analyses with different tie
structure.

### 6. IPTW placebo test script bug (found and fixed, NOT a bug in the real analysis)

The first IPTW placebo attempt (`06b_placebo_iptw.py`) incorrectly reused
the REAL propensity score (fit to predict actual flight exposure) to weight
a random 50/50 pseudo-split. Since pseudo-assignment is independent of the
real propensity score, the stabilized-weight formulas pull in opposite
directions on that score and manufacture an artificial "lift" between two
statistically identical groups (measured -1.93pp, clearly wrong).

This is a flaw in the *placebo test's construction*, not in the real IPTW
analysis -- the real analysis correctly uses a propensity score fit to
predict actual exposure, which is exactly what stabilized IPTW weights are
supposed to invert. **Fix** (`06c_placebo_iptw_fixed.py`): refit a fresh
propensity model against the pseudo-treatment label itself (random by
construction; the refit model's AUC came out to 0.502, confirming it
learned no real signal, as expected) before applying IPTW to the pseudo
groups. With this fix, the IPTW placebo test passes cleanly (lift +0.05pp,
p=0.068).

## Divergence resolved: matched (9.2%) vs. IPTW (21.6%) is real effect heterogeneity, not bias (see `08`/`09`/`10`)

Three diagnostics were run to investigate the gap flagged above:

1. **`08_population_scope_check.py`** -- restricted IPTW to only the matched
   population. This did NOT converge the two estimates (it moved the IPTW
   estimate further away, to 1.7%), ruling out "different population scope"
   as a simple explanation. It did reveal that matched-eligible controls have
   much higher propensity (median 0.34) than the full control pool (median
   0.23) -- expected, since matching only pulls controls that resemble
   treated units.

2. **`09_ps_stratified_lift.py`** -- broke lift down by propensity-score
   decile. Found a strong, **monotonic** pattern: relative lift is +90.1% in
   the lowest-propensity decile and declines steadily to -2.9% in the
   highest-propensity decile. Reconstructing each method's implicit
   weighting from the deciles reproduces both original estimates almost
   exactly: treated-population-weighted average -> 10.6% (~ matches
   matching's 9.2%), equal-weighted average across deciles -> 21.2% (~
   matches IPTW's 21.6%). **This is genuine effect heterogeneity**: the
   campaign has a large incremental effect on users unlikely to be reached
   organically (low propensity), and little-to-no incremental effect on
   users who were likely to be reached/convert anyway (high propensity,
   probably already high-intent). Matching and IPTW are both "correct" --
   they are computing different weighted averages over a genuinely
   heterogeneous treatment effect, not one being more biased than the other.

3. **`10_aipw_estimator.py`** -- fit a doubly-robust AIPW estimator (5-fold
   cross-fitted HistGradientBoostingClassifier outcome models + the existing
   propensity scores) as an independent third estimate. Result: **+15.0%
   relative lift, 95% CI [+11.2%, +18.8%]** (absolute lift 1.24pp, 95% CI
   [0.92pp, 1.55pp]) -- landing cleanly **between** matching and IPTW, exactly
   as expected if the gap is heterogeneity rather than bias in either method.

### Recommended reporting

Report **AIPW (+15.0%, 95% CI [+11.2%, +18.8%])** as the headline
population-average lift number -- it is doubly robust (consistent if either
the propensity or outcome model is correct) and uses both the propensity and
outcome information rather than relying on one weighting scheme. Pair it
with the decile breakdown (`results/lift_by_ps_decile.csv`) to communicate
the heterogeneity story: the campaign is most incremental for
hard-to-reach/low-propensity users, with diminishing returns for
high-propensity users who were likely to convert regardless of exposure.
This is a stronger and more actionable finding than any single point
estimate -- it has direct implications for targeting strategy (e.g. bidding
more aggressively on lower-propensity-to-be-exposed segments where
incremental lift is highest).

## Known limitations / open questions

- Feature set covers only a subset of available device/behavioral
  attributes (device activity, historical ad exposure, demographics, TV
  usage). Richer features (e.g. ACR content genre affinity, HI2A category
  scores) available in `conv_update_union`/`AB_ML_PSID_AGG_WITHOUT_PII`
  were not included in this iteration -- could sharpen the outcome model
  used in AIPW and the propensity model used everywhere.
- Attribution window for `CONVERTED_POST_EXPOSURE` (i.e. how many days
  post-exposure a conversion is still attributed) has not been independently
  confirmed against Samsung Ads' standard attribution policy for conversion
  group 5534.
- The heterogeneity finding (decile 09) has not yet been decomposed by
  *which* covariates drive the low-propensity/high-lift segment (e.g. is it
  a particular device type, geography, or historical engagement tier?) --
  worth a follow-up subgroup analysis if this is used to inform targeting
  changes.

