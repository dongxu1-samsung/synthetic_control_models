# Dimensional Lift Analysis — Hulu Flight 648768 (Campaign 291260)

## Purpose
Product team ask: identify which user segments respond most/least to Hulu ad exposure, to inform campaign targeting and budget optimization.

## Methodology
- **Primary estimator**: mean of the AIPW (doubly-robust) pseudo-outcome per segment, computed on the **full 5.96M population** — unbiased under correct specification of either the propensity or outcome model, and doesn't require re-matching/re-weighting per segment.
- **Cross-check**: mean CATE from the CausalForestDML fit (750K subsample) for the same segment definitions — used to confirm findings aren't an artifact of one method.
- Every segment estimate includes SE, 95% CI, and p-value; segments with n < 2,000 are flagged `low_confidence` and excluded from headline conclusions (mostly tiny international-country buckets with 1-20 users).
- **2D interaction analysis** run on the two strongest single-dimension findings to rule out confounding.

## Headline Findings

### 1. Prior app engagement is the strongest driver — inverse relationship
| Engagement tier | AIPW lift | Rel. lift | CausalForest cross-check | n |
|---|---|---|---|---|
| **No_App_Activity** | **+5.15pp** | **+132.5%** | +4.88pp | 436,475 |
| Champion | +3.35pp | +103.7% | +3.15pp | 160,386 |
| Regular | +3.09pp | +120.5% | +3.04pp | 904,745 |
| Light | +2.97pp | +90.3% | +2.95pp | 4,462,471 |

Users with **zero prior app activity** show ~73% higher relative lift than the largest segment (Light engagement, 75% of the population). AIPW and CausalForestDML agree closely on every tier — this is a robust, real effect, not noise. Classic "diminishing returns on already-warm audiences": already-engaged users convert with or without this specific exposure, so the *incremental* effect is smaller for them.

### 2. Prior mobile-channel ad exposure predicts much lower incremental lift — independent of engagement
| Prior dominant campaign channel | AIPW lift | Rel. lift | n |
|---|---|---|---|
| None | +3.60pp | +110.6% | 1,209,265 |
| Web | +3.50pp | +112.0% | 2,292,771 |
| TV | +3.42pp | +102.3% | 1,011,432 |
| **Mobile** | **+2.05pp** | **+63.2%** | 1,450,609 |

**Verified independent of engagement tier** (interaction check): the Mobile gap holds within every engagement tier —

| Engagement tier | Mobile lift | Non-Mobile avg lift |
|---|---|---|
| No_App_Activity | 3.26pp | ~5.4pp |
| Champion | 2.53pp | ~3.5pp |
| Regular | 1.82pp | ~3.5pp |
| Light | 2.00pp | ~3.3pp |

...and independent of impression-volume quartile too (Mobile is the lowest-lift group in every quartile of `HIST_IMPRESSIONS_7D`). This is a genuinely separate signal, not a proxy for something else we're already capturing.

**Budget context**: ~21% of delivered campaign impressions currently go to Mobile-history users, despite them converting at roughly half the relative lift of every other channel-history segment.

### 3. Historical ad-exposure recency: 0-days-active shows the highest lift
| Recency bucket | AIPW lift | Rel. lift | n |
|---|---|---|---|
| **0_days** (no recent activity) | **+3.96pp** | **+114.1%** | 785,869 |
| 4-7 days | +3.25pp | +102.7% | 3,338,667 |
| 1 day | +2.66pp | +84.7% | 579,534 |
| 2-3 days | +2.63pp | +81.4% | 1,260,007 |

Non-monotonic (0-days highest, 1-3 days lowest, 4-7 days recovers) — likely overlapping with the engagement-tier effect (0-days-active correlates with No_App_Activity). Directionally consistent with finding #1 but weaker as a standalone signal; treat as secondary evidence rather than an independent lever.

### 4. Historical impression volume: monotonic decline then recovery at the top
| Quartile | AIPW lift | Rel. lift |
|---|---|---|
| Q4 (highest) | +3.78pp | +120.6% |
| Q1 (lowest) | +3.28pp | +99.8% |
| Q3 | +2.93pp | +90.6% |
| Q2 | +2.63pp | +81.9% |

U-shaped: both very-low and very-high prior impression-volume users show above-average lift; the middle two quartiles lag. Interacting with campaign type shows this pattern holds across channels (Mobile is consistently lowest at every quartile level).

### 5. Geography: limited actionable signal
- Full US vs UNKNOWN-country split shows a modest gap (UNKNOWN +3.76pp/+113.0% vs US +3.04pp/+95.1%), but "UNKNOWN" is a data-quality bucket, not an actionable targeting dimension.
- Top-10 DMA codes by volume range from +2.55pp to +3.82pp lift (rel. lift 83%-124%) — real variation but no obvious geographic pattern; DMA 511 and 807 show the highest lift, DMA 623 and 528 the lowest. No DMA-name lookup table was available in UDW to translate codes to market names for this write-up.
- Device language: English (94.8% rel. lift) vs Spanish (111.5% rel. lift) show a real, statistically significant gap (both large-n, tight CIs), but this likely reflects a smaller/differently-composed Hispanic-market audience rather than a targetable lever on its own.

## Recommendations

1. **Prioritize budget toward users with no/low recent app engagement.** This is the strongest, most consistent, most cross-validated signal in the analysis (AIPW and CausalForestDML agree almost exactly). Currently this segment (`No_App_Activity`) receives only 6.8% of delivered impressions despite showing the highest per-user incremental lift — there is real headroom to reallocate spend here without diminishing returns from over-targeting an already-small pool (7.3% of the eligible population).

2. **De-prioritize (or apply a lift-adjusted bid discount to) users whose recent ad history is Mobile-dominant.** This is the most actionable, well-isolated finding: independent of engagement tier and impression volume, Mobile-history users convert incrementally at roughly half the rate of every other channel-history group, and currently absorb ~21% of delivered impressions. Shifting even a modest share of that budget toward Web/TV/None-history users (all ~102-112% relative lift) should improve overall campaign efficiency without needing new audience data.

3. **Do not over-index on recency or impression-volume buckets as standalone levers** — both show real but secondary effects that substantially overlap with the engagement-tier finding. Use them as tie-breakers within the engagement-tier segmentation, not as primary targeting axes.

4. **Geography (DMA/country/language) shows statistically real but not obviously actionable variation.** Worth a follow-up with a DMA-name lookup (not available in the current UDW schema) if the product team wants market-level targeting; as-is, the country/language splits likely reflect underlying market-mix differences rather something to target on directly.

5. **Suggested test**: a small controlled budget shift (e.g., 10% of Mobile-history impression budget reallocated to No_App_Activity + Web/TV-history users) with a follow-up lift re-measurement would directly validate whether these dimensional findings translate into an improved blended campaign lift, before committing to a larger reallocation.

## Statistical caveats
- All findings above are from large segments (n > 70K in every case cited) with tight 95% CIs and p < 0.001 — not noise.
- Small international-country buckets (n < 1,000, several with n < 20) were excluded from conclusions; their point estimates are unreliable (see `dimensional_lift_hulu.csv`, `low_confidence=True` rows).
- The AIPW estimator is doubly-robust but still relies on the same covariate set as the rest of this pipeline; unmeasured confounders specific to any one segment (e.g., a market-specific promotional overlap) can't be ruled out from observational data alone.

## Files
- `results_hulu/dimensional_lift_hulu.csv` — full segment-level lift table across 8 dimensions (54 rows)
- `results_hulu/dimensional_lift_interactions_hulu.csv` — 2D interaction tables (engagement × channel, engagement × recency, channel × impression volume)
- `scripts/12_dimensional_lift_hulu.py`, `scripts/12b_interaction_analysis_hulu.py` — analysis code
