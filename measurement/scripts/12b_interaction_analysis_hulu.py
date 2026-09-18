"""
Step 12b: 2D interaction analysis for dimensional lift (Hulu campaign
291260, flight 648768).

Follows up on 12_dimensional_lift_hulu.py's single-dimension findings:
  - ENGAGEMENT_TIER: No_App_Activity shows the highest lift (+5.15pp,
    +132.5% rel.), Light shows the lowest (+2.97pp, +90.3% rel.)
  - DOMINANT_PRIOR_CAMPAIGN_TYPE: Mobile shows a much lower lift (+2.05pp,
    +63.2% rel.) than Web/TV/None (all ~+3.4-3.6pp, ~102-112% rel.)

This script checks whether these two effects are INDEPENDENT (additive) or
whether one is confounding/masking the other -- e.g. is "Mobile" simply
correlated with "Light engagement" (so we're seeing the same underlying
segment twice), or is prior-campaign-channel a genuinely separate driver
of incremental lift on top of engagement tier?
"""
import numpy as np
import pandas as pd
from scipy import stats

AIPW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/aipw_scored_hulu.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/dimensional_lift_interactions_hulu.csv"

MIN_SEGMENT_N = 2000

df = pd.read_parquet(AIPW_PATH)


def engagement_tier(row):
    counts = {"Champion": row["CHAMPION_APP_COUNT"], "Regular": row["REGULAR_APP_COUNT"], "Light": row["LIGHT_APP_COUNT"]}
    if sum(counts.values()) == 0:
        return "No_App_Activity"
    return max(counts, key=counts.get)


def dominant_campaign_type(row):
    counts = {
        "Mobile": row["HIST_MOBILE_CAMPAIGN_IMP_7D"],
        "Web": row["HIST_WEB_CAMPAIGN_IMP_7D"],
        "TV": row["HIST_TV_CAMPAIGN_IMP_7D"],
    }
    if sum(counts.values()) == 0:
        return "None"
    return max(counts, key=counts.get)


df["ENGAGEMENT_TIER"] = df.apply(engagement_tier, axis=1)
df["DOMINANT_PRIOR_CAMPAIGN_TYPE"] = df.apply(dominant_campaign_type, axis=1)
df["HIST_ACTIVE_DAYS_7D_BUCKET"] = pd.cut(
    df["HIST_ACTIVE_DAYS_7D"], bins=[-0.1, 0, 1, 3, 7], labels=["0_days", "1_day", "2-3_days", "4-7_days"]
)


def interaction_table(df, col_a, col_b, min_n=MIN_SEGMENT_N):
    rows = []
    for (val_a, val_b), g in df.groupby([col_a, col_b], observed=True):
        n = len(g)
        if n < min_n:
            continue
        psi = g["AIPW_PSI"].values
        ate = psi.mean()
        se = psi.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = ate - 1.96 * se, ate + 1.96 * se
        rows.append({
            col_a: val_a, col_b: val_b, "n": n,
            "aipw_ate_pp": 100 * ate,
            "aipw_ci_95_lo_pp": 100 * ci_lo,
            "aipw_ci_95_hi_pp": 100 * ci_hi,
        })
    return pd.DataFrame(rows).sort_values("aipw_ate_pp", ascending=False).reset_index(drop=True)


print("=" * 100)
print("INTERACTION: ENGAGEMENT_TIER x DOMINANT_PRIOR_CAMPAIGN_TYPE")
print("=" * 100)
print("Crosstab (row %):")
ct = pd.crosstab(df["ENGAGEMENT_TIER"], df["DOMINANT_PRIOR_CAMPAIGN_TYPE"], normalize="index")
print((ct * 100).round(1).to_string())

inter1 = interaction_table(df, "ENGAGEMENT_TIER", "DOMINANT_PRIOR_CAMPAIGN_TYPE")
print("\nLift by (engagement tier, prior campaign type):")
print(inter1.round(4).to_string(index=False))

print("\n" + "=" * 100)
print("INTERACTION: ENGAGEMENT_TIER x HIST_ACTIVE_DAYS_7D_BUCKET")
print("=" * 100)
print("Crosstab (row %):")
ct2 = pd.crosstab(df["ENGAGEMENT_TIER"], df["HIST_ACTIVE_DAYS_7D_BUCKET"], normalize="index")
print((ct2 * 100).round(1).to_string())

inter2 = interaction_table(df, "ENGAGEMENT_TIER", "HIST_ACTIVE_DAYS_7D_BUCKET")
print("\nLift by (engagement tier, recency bucket):")
print(inter2.round(4).to_string(index=False))

print("\n" + "=" * 100)
print("INTERACTION: DOMINANT_PRIOR_CAMPAIGN_TYPE x HIST_IMPRESSIONS_7D_QUARTILE")
print("=" * 100)
df["HIST_IMPRESSIONS_7D_QUARTILE"] = pd.qcut(
    df["HIST_IMPRESSIONS_7D"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
inter3 = interaction_table(df, "DOMINANT_PRIOR_CAMPAIGN_TYPE", "HIST_IMPRESSIONS_7D_QUARTILE")
print("\nLift by (prior campaign type, impression-volume quartile):")
print(inter3.round(4).to_string(index=False))

# ------------------------------------------------------------------
# Does "Mobile" just mean "Light engagement + low prior impressions" (i.e.
# is the Mobile finding confounded), or does the Mobile effect persist
# WITHIN each engagement tier (i.e. an independent driver)?
# ------------------------------------------------------------------
print("\n" + "=" * 100)
print("KEY CHECK: does the Mobile-channel lift gap persist WITHIN each engagement tier?")
print("=" * 100)
mobile_check = inter1[inter1["ENGAGEMENT_TIER"].isin(["No_App_Activity", "Champion", "Regular", "Light"])]
mobile_check_pivot = mobile_check.pivot(index="ENGAGEMENT_TIER", columns="DOMINANT_PRIOR_CAMPAIGN_TYPE", values="aipw_ate_pp")
print(mobile_check_pivot.round(3).to_string())

all_interactions = pd.concat([
    inter1.assign(interaction="ENGAGEMENT_TIER_x_CAMPAIGN_TYPE"),
    inter2.assign(interaction="ENGAGEMENT_TIER_x_RECENCY"),
    inter3.assign(interaction="CAMPAIGN_TYPE_x_IMPRESSION_QUARTILE"),
], ignore_index=True)
all_interactions.to_csv(OUT_PATH, index=False)
print(f"\nSaved interaction tables to {OUT_PATH}")
