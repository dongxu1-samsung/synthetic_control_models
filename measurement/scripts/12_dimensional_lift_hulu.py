"""
Step 12: Dimensional lift analysis -- "which user segments respond most to
Hulu ad exposure" (Hulu campaign 291260, flight 648768).

Product team ask: identify which user dimensions/segments show the
strongest (and weakest) incremental lift from ad exposure, to inform
campaign targeting/budget optimization.

Methodology:
  - PRIMARY estimator: mean of the AIPW pseudo-outcome (psi_i) within each
    segment. This is unbiased for the segment's ATE under either correct
    propensity or correct outcome specification (doubly-robust), computed
    on the FULL 5.96M population (not a subsample) -- this is the most
    statistically efficient and least-biased per-segment estimate we have
    available, since it doesn't require re-matching or re-weighting per
    segment.
  - SECONDARY cross-check: mean CATE from the CausalForestDML fit (750K
    subsample, 09_causal_forest_hulu.py) for the same segment definitions.
    Agreement between AIPW-segment-means and CATE-segment-means is strong
    evidence the finding is real rather than an artifact of either method.
  - Naive (unadjusted) rates are also reported per segment for reference,
    to show where selection bias is inflating/deflating a segment's naive
    lift-looking number.
  - Every segment lift estimate includes a standard error (psi_i std / sqrt(n)
    within the segment) and a two-sided p-value / 95% CI, so small segments
    with noisy estimates can be flagged rather than over-interpreted.
"""
import numpy as np
import pandas as pd
from scipy import stats

AIPW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/aipw_scored_hulu.parquet"
CATE_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/cate_scored_hulu.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/dimensional_lift_hulu.csv"

MIN_SEGMENT_N = 2000  # below this, flag as "low-confidence" rather than drop

df = pd.read_parquet(AIPW_PATH)
cate_df = pd.read_parquet(CATE_PATH)[["PSID", "CATE"]]
print(f"AIPW-scored population: {df.shape}")
print(f"CausalForestDML subsample (for cross-check): {cate_df.shape}")

overall_ate = df["AIPW_PSI"].mean()
overall_control_rate = df.loc[df.EXPOSED == 0, "CONVERTED_POST_EXPOSURE"].mean()
overall_rel_lift = 100 * overall_ate / overall_control_rate
print(f"\nOverall AIPW ATE: {overall_ate:.5f} ({100*overall_ate:.3f}pp), "
      f"rel lift {overall_rel_lift:.2f}% (baseline: matching 91.5%, IPTW 102.3%, "
      f"causal forest 96.7%)")


def segment_lift(df, group_col, min_n=MIN_SEGMENT_N):
    """Per-segment AIPW ATE (mean psi), naive lift, SE, CI, p-value, and n."""
    rows = []
    for val, g in df.groupby(group_col, observed=True):
        n = len(g)
        psi = g["AIPW_PSI"].values
        ate = psi.mean()
        se = psi.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ci_lo, ci_hi = ate - 1.96 * se, ate + 1.96 * se
        z = ate / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

        naive_t = g.loc[g.EXPOSED == 1, "CONVERTED_POST_EXPOSURE"].mean()
        naive_c = g.loc[g.EXPOSED == 0, "CONVERTED_POST_EXPOSURE"].mean()
        naive_lift = naive_t - naive_c

        rows.append({
            "segment_value": val,
            "n": n,
            "exposed_rate": g["EXPOSED"].mean(),
            "aipw_ate_pp": 100 * ate,
            "aipw_se_pp": 100 * se,
            "aipw_ci_95_lo_pp": 100 * ci_lo,
            "aipw_ci_95_hi_pp": 100 * ci_hi,
            "aipw_p_value": p,
            "naive_lift_pp": 100 * naive_lift,
            "control_conv_rate_pct": 100 * naive_c,
            "rel_lift_pct_vs_control": 100 * ate / naive_c if naive_c > 0 else np.nan,
            "low_confidence": n < min_n,
        })
    out = pd.DataFrame(rows).sort_values("aipw_ate_pp", ascending=False).reset_index(drop=True)
    return out


def cate_cross_check(cate_df_full, group_col_series_full, group_col_name):
    """Mean CATE per segment from the causal forest subsample, for cross-checking."""
    tmp = cate_df_full.copy()
    tmp[group_col_name] = group_col_series_full
    g = tmp.groupby(group_col_name, observed=True)["CATE"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "cate_forest_mean_pp", "count": "cate_forest_n"})
    g["cate_forest_mean_pp"] = 100 * g["cate_forest_mean_pp"]
    g = g.rename(columns={group_col_name: "segment_value"})
    return g


all_results = []

# ------------------------------------------------------------------
# Dimension 1: Device country
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: DEVICE_COUNTRY")
print("=" * 100)
seg = segment_lift(df, "DEVICE_COUNTRY")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DEVICE_COUNTRY")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 2: App engagement tier (dominant lifecycle category)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: ENGAGEMENT_TIER (dominant app lifecycle category)")
print("=" * 100)
def engagement_tier(row):
    counts = {"Champion": row["CHAMPION_APP_COUNT"], "Regular": row["REGULAR_APP_COUNT"], "Light": row["LIGHT_APP_COUNT"]}
    if sum(counts.values()) == 0:
        return "No_App_Activity"
    return max(counts, key=counts.get)

df["ENGAGEMENT_TIER"] = df.apply(engagement_tier, axis=1)
seg = segment_lift(df, "ENGAGEMENT_TIER")
print(seg.round(4).to_string(index=False))
cate_engagement_tier = cate_df.merge(
    df[["PSID", "ENGAGEMENT_TIER"]], on="PSID", how="inner"
)
cc = cate_cross_check(cate_engagement_tier, cate_engagement_tier["ENGAGEMENT_TIER"], "ENGAGEMENT_TIER")
seg = seg.merge(cc, on="segment_value", how="left")
print("\nCross-check (CausalForestDML subsample):")
print(cc.round(4).to_string(index=False))
seg.insert(0, "dimension", "ENGAGEMENT_TIER")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 3: Historical ad-exposure recency (active days in pre-period)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: HIST_ACTIVE_DAYS_7D_BUCKET (prior ad-exposure recency)")
print("=" * 100)
df["HIST_ACTIVE_DAYS_7D_BUCKET"] = pd.cut(
    df["HIST_ACTIVE_DAYS_7D"], bins=[-0.1, 0, 1, 3, 7], labels=["0_days", "1_day", "2-3_days", "4-7_days"]
)
seg = segment_lift(df, "HIST_ACTIVE_DAYS_7D_BUCKET")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "HIST_ACTIVE_DAYS_7D_BUCKET")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 4: Historical ad impression volume (pre-period, quartiled)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: HIST_IMPRESSIONS_7D_QUARTILE (prior ad exposure volume)")
print("=" * 100)
df["HIST_IMPRESSIONS_7D_QUARTILE"] = pd.qcut(
    df["HIST_IMPRESSIONS_7D"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
seg = segment_lift(df, "HIST_IMPRESSIONS_7D_QUARTILE")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "HIST_IMPRESSIONS_7D_QUARTILE")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 5: TV age proxy (STV_YEAR, binned)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: STV_YEAR_BUCKET (device/TV age proxy)")
print("=" * 100)
df["STV_YEAR_BUCKET"] = pd.cut(
    df["STV_YEAR"], bins=[2015, 2019, 2021, 2023, 2026], labels=["2016-2019", "2020-2021", "2022-2023", "2024-2026"]
)
seg = segment_lift(df, "STV_YEAR_BUCKET")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "STV_YEAR_BUCKET")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 6: Device language
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: DEVICE_LANGUAGE (top 5 by volume + OTHER)")
print("=" * 100)
lang_counts = df["DEVICE_LANGUAGE"].value_counts()
top_langs = lang_counts.head(5).index.tolist()
df["DEVICE_LANGUAGE_GROUPED"] = df["DEVICE_LANGUAGE"].astype(str)
df.loc[~df["DEVICE_LANGUAGE_GROUPED"].isin(top_langs), "DEVICE_LANGUAGE_GROUPED"] = "OTHER"
seg = segment_lift(df, "DEVICE_LANGUAGE_GROUPED")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DEVICE_LANGUAGE_GROUPED")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 7: Top DMAs by volume (numeric codes -- no name lookup table
# available in UDW; still directly actionable since campaign targeting
# tools operate on DMA codes)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: DMA_CODE (top 10 by volume)")
print("=" * 100)
dma_counts = df["DMA_CODE"].value_counts()
top_dmas = dma_counts.head(10).index.tolist()
df["DMA_TOP10"] = df["DMA_CODE"].where(df["DMA_CODE"].isin(top_dmas), other=np.nan)
seg = segment_lift(df[df["DMA_TOP10"].notna()], "DMA_TOP10")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DMA_CODE_TOP10")
all_results.append(seg)

# ------------------------------------------------------------------
# Dimension 8: Prior campaign-type exposure mix (mobile/web/TV)
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: DOMINANT_PRIOR_CAMPAIGN_TYPE (mobile/web/TV/none)")
print("=" * 100)
def dominant_campaign_type(row):
    counts = {
        "Mobile": row["HIST_MOBILE_CAMPAIGN_IMP_7D"],
        "Web": row["HIST_WEB_CAMPAIGN_IMP_7D"],
        "TV": row["HIST_TV_CAMPAIGN_IMP_7D"],
    }
    if sum(counts.values()) == 0:
        return "None"
    return max(counts, key=counts.get)

df["DOMINANT_PRIOR_CAMPAIGN_TYPE"] = df.apply(dominant_campaign_type, axis=1)
seg = segment_lift(df, "DOMINANT_PRIOR_CAMPAIGN_TYPE")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DOMINANT_PRIOR_CAMPAIGN_TYPE")
all_results.append(seg)

# ------------------------------------------------------------------
# Consolidate
# ------------------------------------------------------------------
combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved consolidated dimensional lift table to {OUT_PATH}")
print(f"Total rows: {len(combined)}, dimensions covered: {combined['dimension'].nunique()}")
