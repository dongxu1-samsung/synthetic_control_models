"""
Step 20: Dimensional lift analysis on the MERGED Apple dataset.

Product team ask: identify which user dimensions/segments show the
strongest (and weakest) incremental lift from ad exposure, to inform
campaign targeting/budget optimization -- now using the richer merged
feature set (device/geo from S3, ACR app-category usage, ad-exposure
diversity, TV usage day-part ratios) in addition to the original
engagement/recency/device dimensions.

Methodology: same as 12_dimensional_lift_hulu.py -- primary estimator is
the mean AIPW pseudo-outcome (psi_i) within each segment (doubly-robust,
computed on the full population, no re-matching/re-weighting needed per
segment). Naive (unadjusted) rates reported for reference/bias comparison.
Segments below MIN_SEGMENT_N are flagged low-confidence rather than
dropped.
"""
import numpy as np
import pandas as pd
from scipy import stats

AIPW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_aipw_scored.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_dimensional_lift.csv"

MIN_SEGMENT_N = 2000

df = pd.read_parquet(AIPW_PATH)
print(f"AIPW-scored population: {df.shape}")

# The raw AIPW pseudo-outcome (psi_i) has a small number of extreme values
# driven by observations with near-0/near-1 propensity scores in the
# denominator (0.10% of rows have |psi| > 10, some as large as +/-9700),
# making the unwinsorized ATE unstable. We initially tried winsorizing
# psi_i directly (clipping at 1st/99th percentile) but this BACKFIRED: the
# psi distribution is asymmetric, so symmetric-percentile clipping of the
# VALUES shifts the mean substantially in one direction (winsorized overall
# ATE came out -25% relative lift, wildly inconsistent with the sensitivity
# check across common-support trim widths, which is stable and *positive*
# in the +0.4%..+2.7% range as trim widens -- see /tmp/aipw_trim_sensitivity.py).
#
# The correct, standard fix is instead to TRIM ON THE PROPENSITY SCORE
# itself (drop rows with extreme predicted P(exposed), i.e. classic
# common-support restriction) rather than clip the resulting outcome
# pseudo-value. This is principled (removes units where neither nuisance
# model is well-identified) rather than a post-hoc distortion of the
# analysis population's average outcome.
ps_lo, ps_hi = df["PROPENSITY_SCORE"].quantile([0.01, 0.99])
before_n = len(df)
df = df[(df["PROPENSITY_SCORE"] >= ps_lo) & (df["PROPENSITY_SCORE"] <= ps_hi)].copy()
print(f"Trimmed to propensity-score common support [{ps_lo:.4f}, {ps_hi:.4f}] (1st/99th pctile): "
      f"{len(df)} / {before_n} rows retained ({100*len(df)/before_n:.1f}%)")

overall_ate = df["AIPW_TERM"].mean()
overall_control_rate = df.loc[df.EXPOSED == 0, "CONVERTED_POST_EXPOSURE"].mean()
overall_rel_lift = 100 * overall_ate / overall_control_rate
print(f"\nOverall AIPW ATE: {overall_ate:.5f} ({100*overall_ate:.3f}pp), rel lift {overall_rel_lift:.2f}%")


def segment_lift(df, group_col, min_n=MIN_SEGMENT_N):
    rows = []
    for val, g in df.groupby(group_col, observed=True):
        n = len(g)
        psi = g["AIPW_TERM"].values
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


all_results = []

print("\n" + "=" * 100)
print("DIMENSION: DEVICE_COUNTRY")
print("=" * 100)
seg = segment_lift(df, "DEVICE_COUNTRY")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DEVICE_COUNTRY")
all_results.append(seg)

print("\n" + "=" * 100)
print("DIMENSION: OPERATING_SYSTEM (Tizen TV OS version)")
print("=" * 100)
seg = segment_lift(df, "OPERATING_SYSTEM")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "OPERATING_SYSTEM")
all_results.append(seg)

print("\n" + "=" * 100)
print("DIMENSION: NETSPEED (connection type)")
print("=" * 100)
seg = segment_lift(df, "NETSPEED")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "NETSPEED")
all_results.append(seg)

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
seg.insert(0, "dimension", "ENGAGEMENT_TIER")
all_results.append(seg)

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
# New dimensions enabled by the S3 dataset
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("DIMENSION: DOMINANT_ACR_APP_CATEGORY (education/lifestyle/information/videos/sports)")
print("=" * 100)
def dominant_acr_category(row):
    counts = {
        "Education": row["ACR_APP_EDUCATION"], "Lifestyle": row["ACR_APP_LIFESTYLE"],
        "Information": row["ACR_APP_INFORMATION"], "Videos": row["ACR_APP_VIDEOS"],
        "Sports": row["ACR_APP_SPORTS"],
    }
    if sum(counts.values()) == 0:
        return "No_ACR_Activity"
    return max(counts, key=counts.get)

df["DOMINANT_ACR_APP_CATEGORY"] = df.apply(dominant_acr_category, axis=1)
seg = segment_lift(df, "DOMINANT_ACR_APP_CATEGORY")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "DOMINANT_ACR_APP_CATEGORY")
all_results.append(seg)

print("\n" + "=" * 100)
print("DIMENSION: AD_EXPOSURE_DIVERSITY_QUARTILE (AD_CATEGORY_COUNT, prior ad-category breadth)")
print("=" * 100)
df["AD_EXPOSURE_DIVERSITY_QUARTILE"] = pd.qcut(
    df["AD_CATEGORY_COUNT"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
seg = segment_lift(df, "AD_EXPOSURE_DIVERSITY_QUARTILE")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "AD_EXPOSURE_DIVERSITY_QUARTILE")
all_results.append(seg)

print("\n" + "=" * 100)
print("DIMENSION: AVG_DAILY_HOURS_WATCHED_QUARTILE (overall TV viewing intensity)")
print("=" * 100)
df["AVG_DAILY_HOURS_WATCHED_QUARTILE"] = pd.qcut(
    df["AVG_DAILY_HOURS_WATCHED"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
seg = segment_lift(df, "AVG_DAILY_HOURS_WATCHED_QUARTILE")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "AVG_DAILY_HOURS_WATCHED_QUARTILE")
all_results.append(seg)

print("\n" + "=" * 100)
print("DIMENSION: NUM_UNIQUE_APPS_QUARTILE (breadth of app usage)")
print("=" * 100)
df["NUM_UNIQUE_APPS_QUARTILE"] = pd.qcut(
    df["NUM_UNIQUE_APPS"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
seg = segment_lift(df, "NUM_UNIQUE_APPS_QUARTILE")
print(seg.round(4).to_string(index=False))
seg.insert(0, "dimension", "NUM_UNIQUE_APPS_QUARTILE")
all_results.append(seg)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved consolidated dimensional lift table to {OUT_PATH}")
print(f"Total rows: {len(combined)}, dimensions covered: {combined['dimension'].nunique()}")
