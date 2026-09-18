# Synthetic Control Measurement -- MERGED Dataset Pipeline (Apple TV, Flight 658095)

Campaign 296636, Flight 658095, Conversion Group 5534 (Apple TV app, App ID
3201807016597). This extends the original pipeline (`README.md`,
`scripts/01-10`) by merging in a teammate-provided feature export from S3
(`s3://pi-expt-use1-dev/.../scm_dataset_658095_2026-08-05_2026-08-19_US.csv`,
5,708,675 rows x 63 columns) and re-running the full propensity -> matching/
IPTW -> balance -> lift -> AIPW -> dimensional-CATE pipeline on the combined,
deduplicated feature set.

## Pipeline (run in order from `measurement/`)

11. `11_merge_datasets.py` -- inner-join our `PROPENSITY_FEATURES_APPLE`
    (5,742,285 rows) with the S3 export on PSID, after validating population
    overlap (99.25%/99.84%) and label agreement (EXPOSED, CONVERTED_POST_
    EXPOSURE, CAMPAIGN_IMPRESSIONS all 100% match on the overlap). Drops 11
    S3 columns that are exact-duplicate labels or >99.97% constant. Result:
    5,699,320 rows x 99 columns.
12. `12_feature_analysis.py` -- missingness, zero-variance check, categorical
    cardinality, and pairwise correlation scan (|corr| > 0.90) across all 84
    numeric features in the merged set. Flags 28 highly-correlated pairs.
13. `13_final_dedup.py` -- drops a further 11 near-duplicate/zero-variance
    columns identified by the correlation scan (see "Feature dedup" below).
    Result: 5,699,320 rows x 88 columns (85 usable features after labels).
14. `14_fit_propensity_merged.py` -- refits the exposure-propensity logistic
    regression on the full deduped feature set (67 numeric + low-cardinality
    categoricals). **Test AUC improves from 0.7229 (original, ~33 features)
    to 0.7887 (merged, ~82 model inputs)** -- see "Model performance" below.
15. `15_match_and_weight_merged.py` -- common-support trim, 1:1 caliper
    matching (100% of treated matched within caliper 0.01), stabilized/
    trimmed IPTW.
16. `16_balance_diagnostics_merged.py` -- SMD table for 60 covariates
    (original + new). 34/60 raw-imbalanced (max |SMD| 0.65) -> **0/60 above
    threshold after BOTH matching and IPTW**.
17. `17_lift_estimation_merged.py` -- naive / matched / IPTW lift estimates.
18. `18_ps_stratified_lift_merged.py` -- propensity-decile breakdown, needed
    because matched and IPTW estimates diverge (and sign-flip) on the
    merged data -- see "Lift results" below.
19. `19_aipw_estimator_merged.py` -- doubly-robust AIPW estimator (5-fold
    cross-fitted HistGradientBoostingClassifier outcome models), with
    propensity-score common-support trimming (1st/99th pctile) for
    stability -- see "AIPW stability" below.
20. `20_dimensional_lift_merged.py` -- CATE by 14 segment dimensions (device/
    OS/geo, engagement tier, ad-exposure recency/volume/diversity, ACR app-
    category usage, TV-viewing intensity, app-usage breadth), using the
    per-unit AIPW pseudo-outcome as the primary estimator.

Result CSVs are in `results_apple_merged/` (balance table, lift summaries,
decile breakdown, three-way comparison, dimensional lift table, propensity
coefficients, correlation/descriptive-stats scans). Large intermediate
Parquet files (merged features, scored/matched/weighted/AIPW datasets,
~1-2GB each) are kept locally only, not committed to the repo.

## Merge methodology & feature dedup

**Population overlap validation** (before merging): our dataset has
5,742,285 distinct PSIDs, the S3 export has 5,708,675; overlap is 5,699,320
(99.25% of ours, 99.84% of S3's). On the overlap, `EXPOSED`/`exposed`,
`CONVERTED_POST_EXPOSURE`/`converted`, and `CAMPAIGN_IMPRESSIONS`/
`impressions` are **100% identical** (correlation 1.0 for the impression
count) -- strong confirmation both datasets describe the same underlying
population/labels for this flight, built via independent feature
pipelines, and are safe to inner-join as complementary feature sources.

**Columns dropped from the S3 side before merge** (11 total): exact-duplicate
labels (`exposed`, `impressions`, `converted` -- superseded by our
canonical `EXPOSED`/`CAMPAIGN_IMPRESSIONS`/`CONVERTED_POST_EXPOSURE`), and
6 columns that are 99.97-100% a single constant value across the whole S3
file (`samsung_affinity`, `device_type_id`, `uid_type`, `uid_sticky`,
`ratio_time_avod_true`, `ratio_time_avod_false`, plus 5
`bid_request_device_ext_*`/`bid_request_ext_channel` fields that are 99.97%
one value with the 0.03% remainder just an "UNK" bucket carrying no signal).

**Further dedup after merge** (11 more columns, found via the |corr| > 0.90
correlation scan): near-perfect duplicate pairs within either source
(`HIST_IMP_EVENTS_7D`/`HIST_TILE_IMPRESSIONS_7D` vs `HIST_IMPRESSIONS_7D`,
corr ~1.0; `G_USER_ADJUSTED_MEDIA_PRICE`/`G_USER_BUYER_PRICE` vs
`G_USER_MEDIA_PRICE`, corr 0.999+; `G_USER_VIDEO_COMPLETE_SUM` vs
`G_USER_VIDEO_START_SUM`, corr 0.999). **Lesson learned mid-pipeline**:
`AD_COUNT`/`ADVERTISER_COUNT`/`BRAND_COUNT` looked like 3 distinct
dimensions on raw-value correlation (0.94-0.98) but are effectively
duplicates once log1p-transformed for modeling (0.995-0.999, condition
number ~70) -- fitting all three produced unstable, oversized
opposite-signed logistic coefficients (a classic multicollinearity
symptom). Dropped `ADVERTISER_COUNT`/`BRAND_COUNT`, kept `AD_COUNT`.
`CHANNEL_TILE_RATIO`/`DEVICE_COUNTRY_MISSING` correlate at exactly -1.0
(both derived from "had any pre-period impression row") -- kept the
continuous `CHANNEL_TILE_RATIO`. `HIST_CTV_IMPRESSIONS_7D` is zero-variance
(no CTV-tagged impressions for this flight/app).

## Model performance

| | Original (33 features) | Merged (82 model inputs, 85 usable features) |
|---|---|---|
| Train AUC | 0.7219 | 0.7880 |
| Test AUC | 0.7223 | 0.7887 |

The richer S3 feature set (ACR app-category usage, ad-exposure diversity,
day-part/language viewing ratios, ott-level device/geo/connection signals)
meaningfully improves propensity discrimination (+6.6pp AUC). Top-|coefficient|
standardized features in the merged model: `ACR_APP_USED_APP_CNT` (+1.67),
`AD_COUNT` (+0.88), `AD_CATEGORY_COUNT` (-0.76), `TOTAL_ACTIVE_DAYS` (+0.56),
`ACR_APP_INFORMATION` (-0.52), `G_USER_IMPRESSION_21D` (+0.49) -- full table
in `results_apple_merged/merged_propensity_coefficients.csv`.

## Balance diagnostics

34/60 covariates imbalanced raw (unadjusted), max |SMD| 0.65
(`NUM_UNIQUE_APPS`). **After 1:1 caliper matching: 0/60 covariates above the
0.1 SMD threshold (max |SMD| 0.058).** **After stabilized/trimmed IPTW: 0/60
above threshold (max |SMD| 0.094).** Both adjustment methods achieve strong
covariate balance on the full merged feature set. Full table in
`results_apple_merged/merged_balance_table.csv`.

## Lift results

| Method | Conv. rate (test) | Conv. rate (control) | Abs. lift | Rel. lift |
|---|---|---|---|---|
| Naive (unadjusted) | 10.37% | 7.40% | +2.97pp | +40.1% |
| 1:1 caliper matching | 10.37% | 11.23% | **-0.86pp** | **-7.6%** |
| IPTW (stabilized, trimmed) | 9.27% | 8.47% | +0.80pp | **+9.4%** |
| **AIPW (doubly robust, PS-trimmed)** | -- | -- | **+0.09pp** | **+1.0%** (95% CI [+0.01pp, +0.17pp], i.e. [+0.1%, +2.0%] rel.) |

**Matched and IPTW now disagree in SIGN** (-7.6% vs +9.4%), a materially
worse divergence than the original (non-merged) pipeline's 9.2% vs 21.6%.
Root cause, confirmed via propensity-decile stratification
(`18_ps_stratified_lift_merged.py`): the richer feature set gives the
propensity model much stronger discrimination (AUC 0.79 vs 0.72), which
(a) reveals a strongly monotonic decline in lift across the propensity
distribution -- **decile 0 (lowest propensity) shows +92.5% relative lift,
declining to -19.7% in decile 9 (highest propensity)** -- and (b) shifts
disproportionately-high-propensity, disproportionately-high-baseline-
conversion controls into reuse during matching (reuse count correlates
0.47 with propensity score), pulling the matched-control rate up and
flipping the sign. This is genuine effect heterogeneity, not bias in
either method -- see the CATE/dimensional section below for the underlying
segment-level explanation.

**Headline recommendation: AIPW, +1.0% relative lift (95% CI [+0.1%,
+2.0%]), i.e. a small but statistically significant positive effect once
richer confounders are accounted for.** This is dramatically smaller than
either the matched or IPTW point estimate and much smaller than the
original (non-merged) pipeline's AIPW estimate of +15.0%. The most likely
explanation is that the additional S3 features (particularly ACR app usage
and ad-exposure history) capture confounding that was previously absorbed
into the treatment effect -- i.e. the ~15% and ~21.6% earlier estimates
were partly inflated by omitted-variable bias from users who are both
more likely to be exposed AND more likely to convert for reasons unrelated
to this specific campaign. Report the range (naive 40.1% as an upper bound
on the scale of selection bias; AIPW +1.0% as the best point estimate) to
the product team rather than a single confident number, consistent with
prior guidance on this pipeline to avoid false-confidence framing.

## AIPW stability

The AIPW pseudo-outcome (psi_i) has a small fraction of extreme values (0.10%
of rows have |psi_i| > 10, driven by observations with near-0/near-1
propensity scores in the 1/p or 1/(1-p) denominator). The naive full-sample
AIPW ATE is highly sensitive to how much of this tail is included (ranges
+0.4% to +2.7% relative lift across 0.1%-5% common-support trim widths).
**Lesson learned**: winsorizing the resulting psi_i VALUES directly
(clipping at 1st/99th percentile) is the wrong fix here -- the psi
distribution is asymmetric, so symmetric-percentile value-clipping shifted
the mean to -25% relative lift, wildly inconsistent with the trim-width
sensitivity check. The correct, standard fix is trimming on the
**propensity score itself** (drop the 1st/99th-percentile tails of P(exposed),
98.0% of rows retained) before averaging psi_i -- this removes units where
neither nuisance model is well-identified, rather than distorting the
outcome distribution directly. All reported AIPW numbers use this
PS-trimmed estimator.

## Dimensional lift insights (CATE by segment)

Primary estimator: mean AIPW pseudo-outcome (PS-trimmed) within each
segment, computed on the full population. See
`results_apple_merged/merged_dimensional_lift.csv` for the complete table
(14 dimensions, 63 segment rows). Headline findings:

- **Novelty/low-baseline-exposure users respond far more than habituated
  users.** `HIST_ACTIVE_DAYS_7D_BUCKET = 0_days` (no pre-period ad exposure):
  +41.5% relative lift (p<0.0001, n=587K). `4-7_days` (heavily pre-exposed):
  **-8.5%** (p<0.0001, n=3.1M). Same pattern in `HIST_IMPRESSIONS_7D_QUARTILE`:
  Q1 (lowest prior exposure) +24.6%, Q4 (highest) -8.8%.
- **Light/no app engagement users respond more than heavy app users.**
  `ENGAGEMENT_TIER = No_App_Activity`: +46.6% (p<0.0001, n=728K).
  `Light`: -4.7% (p<0.0001, n=4.3M). `Champion`/`Regular` not significant
  (small samples, wide CIs).
- **Lower overall TV-viewing intensity correlates with higher lift.**
  `AVG_DAILY_HOURS_WATCHED_QUARTILE`: Q1 (lightest viewers) +32.8%, Q3/Q4
  (heaviest viewers) -7.8%/-11.1%. Same direction for `NUM_UNIQUE_APPS_QUARTILE`
  (app-usage breadth): Q1 +36.2%, Q4 -9.1%.
- **DEVICE_COUNTRY = UNKNOWN segment (11.5% of population) shows the
  strongest positive lift** (+41.5%, p<0.0001) of any device/geo cut --
  this segment also has the lowest ad-exposure history, consistent with the
  novelty-effect pattern above (likely newer/less-instrumented devices).
- **TIZEN_TV_6 (newer OS, 11.2% of population) shows negative lift**
  (-9.7%, p<0.0001) vs. TIZEN_TV (+3.0%, p<0.0001) -- direction consistent
  with the OS-6 segment's higher pre-period ad exposure/engagement profile.
- **ACR app-category usage**: `No_ACR_Activity` (+74.6%) and `Information`
  (+64.2%) show the strongest lift; `Videos` (the dominant category, 92.7%
  of population) is flat/slightly negative (-0.8%, not significant at the
  overall level once PS-trimmed).
- **DMA-level lift varies from +33.8% (DMA 602) to -20.7% (DMA 807)** --
  actionable for geo-targeting, though individual DMA sample sizes (98K-733K)
  are smaller than the segment-level cuts above.

Cross-checking naive vs. AIPW lift per segment shows the naive numbers are
uniformly upward-biased (by 1-5pp) relative to AIPW across nearly every
segment -- consistent with the overall naive-vs-adjusted gap, and a reminder
that the raw/unadjusted numbers should never be reported to the product
team as segment-level lift.

## Recommendation for product team

1. **Overall campaign lift is small but real: +1.0% relative (AIPW,
   95% CI [+0.1%, +2.0%])** -- treat this as the headline number, not the
   naive 40.1% or either single-method (matched/IPTW) estimate, both of
   which are now shown to diverge in sign once richer confounders are
   modeled.
2. **Target under-exposed, lower-engagement users for the strongest
   incremental lift**: users with zero/low pre-period ad exposure
   (+41.5%/+24.6%), no dominant app-engagement tier (+46.6%), lighter
   overall TV viewers (+32.8%), and the DEVICE_COUNTRY=UNKNOWN /
   No_ACR_Activity segments (+41.5%/+74.6%) are the highest-lift
   sub-populations found in this analysis.
3. **De-prioritize or re-evaluate spend on heavily pre-exposed, high-
   engagement users** (4-7 days prior exposure: -8.5%; Champion/heavy-app
   users trending negative though not always significant) -- ad saturation
   or diminishing returns is the most likely explanation, consistent with
   the propensity-decile pattern (lift declines monotonically from +92.5%
   to -19.7% across the exposure-propensity distribution).
4. **Investigate DMA-level targeting**: an 11 DMAs analyzed span +33.8% to
   -20.7% relative lift; worth a dedicated geo-optimization pass once more
   DMAs reach adequate sample size.
5. Given how sensitive the headline number was to feature richness (15.0%
   -> 1.0% once the S3 features were merged in), **treat any future lift
   number from this pipeline as provisional until a similarly thorough
   feature audit is repeated** -- omitted-variable bias from missing
   engagement/usage confounders is the single largest source of estimate
   instability found across this entire analysis.
