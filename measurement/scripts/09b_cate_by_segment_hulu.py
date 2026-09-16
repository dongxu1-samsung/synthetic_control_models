"""
Step 9b: CATE by segment -- which user segments respond most to Hulu ad
exposure (Hulu campaign 291260, flight 648768).

Business framing: the causal forest (09_causal_forest_hulu.py) gives a
personalized tau(X) for every unit. Aggregating by natural business
segments (device country, DMA, historical engagement tier, prior ad
exposure recency) turns that into an actionable targeting readout: "which
groups should get priority budget allocation."
"""
import numpy as np
import pandas as pd

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/cate_scored_hulu.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/cate_by_segment_hulu.csv"

df = pd.read_parquet(IN_PATH)
print("Loaded CATE-scored subsample:", df.shape)
print(f"Overall mean CATE: {df['CATE'].mean():.5f}")


def segment_summary(df, group_col, min_n=500):
    g = df.groupby(group_col).agg(
        n=("CATE", "size"),
        mean_cate=("CATE", "mean"),
        std_cate=("CATE", "std"),
        mean_ci_lower=("CATE_CI_LOWER", "mean"),
        mean_ci_upper=("CATE_CI_UPPER", "mean"),
        exposed_rate=("EXPOSED", "mean"),
        raw_conv_rate=("CONVERTED_POST_EXPOSURE", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_n].sort_values("mean_cate", ascending=False)
    return g


print("\n" + "=" * 90)
print("CATE by DEVICE_COUNTRY")
print("=" * 90)
seg_country = segment_summary(df, "DEVICE_COUNTRY", min_n=200)
print(seg_country.round(5).to_string(index=False))

print("\n" + "=" * 90)
print("CATE by engagement tier (derived from ACTIVE_APP_COUNT / LIGHT_APP_COUNT split)")
print("=" * 90)
# Reconstruct a simple engagement-tier label from the app-lifecycle counts
# already in the feature table (Champion/Regular/Light dominant category).
def engagement_tier(row):
    counts = {
        "Champion": row["CHAMPION_APP_COUNT"],
        "Regular": row["REGULAR_APP_COUNT"],
        "Light": row["LIGHT_APP_COUNT"],
    }
    if sum(counts.values()) == 0:
        return "No_App_Activity"
    return max(counts, key=counts.get)

df["ENGAGEMENT_TIER"] = df.apply(engagement_tier, axis=1)
seg_engagement = segment_summary(df, "ENGAGEMENT_TIER", min_n=500)
print(seg_engagement.round(5).to_string(index=False))

print("\n" + "=" * 90)
print("CATE by prior ad-exposure recency (HIST_ACTIVE_DAYS_7D bucketed)")
print("=" * 90)
df["HIST_ACTIVE_DAYS_7D_BUCKET"] = pd.cut(
    df["HIST_ACTIVE_DAYS_7D"], bins=[-0.1, 0, 1, 3, 7], labels=["0_days", "1_day", "2-3_days", "4-7_days"]
)
seg_recency = segment_summary(df, "HIST_ACTIVE_DAYS_7D_BUCKET", min_n=500)
print(seg_recency.round(5).to_string(index=False))

print("\n" + "=" * 90)
print("CATE by historical ad impression volume (HIST_IMPRESSIONS_7D quartile)")
print("=" * 90)
df["HIST_IMPRESSIONS_7D_QUARTILE"] = pd.qcut(
    df["HIST_IMPRESSIONS_7D"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
seg_imp_quartile = segment_summary(df, "HIST_IMPRESSIONS_7D_QUARTILE", min_n=500)
print(seg_imp_quartile.round(5).to_string(index=False))

print("\n" + "=" * 90)
print("CATE by STV_YEAR (TV age proxy, binned)")
print("=" * 90)
df["STV_YEAR_BUCKET"] = pd.cut(
    df["STV_YEAR"], bins=[2015, 2019, 2021, 2023, 2026], labels=["2016-2019", "2020-2021", "2022-2023", "2024-2026"]
)
seg_tv_year = segment_summary(df, "STV_YEAR_BUCKET", min_n=500)
print(seg_tv_year.round(5).to_string(index=False))

# Consolidate all segment tables into one CSV with a "segment_dimension" column
all_segments = []
for name, seg_df in [
    ("DEVICE_COUNTRY", seg_country),
    ("ENGAGEMENT_TIER", seg_engagement),
    ("HIST_ACTIVE_DAYS_7D_BUCKET", seg_recency),
    ("HIST_IMPRESSIONS_7D_QUARTILE", seg_imp_quartile),
    ("STV_YEAR_BUCKET", seg_tv_year),
]:
    seg_df = seg_df.rename(columns={seg_df.columns[0]: "segment_value"})
    seg_df.insert(0, "segment_dimension", name)
    all_segments.append(seg_df)

combined = pd.concat(all_segments, ignore_index=True)
combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved consolidated segment CATE table to {OUT_PATH}")

print("\n" + "=" * 90)
print("TOP-LINE TAKEAWAYS")
print("=" * 90)
print(f"Highest-CATE country segment: {seg_country.iloc[0]['DEVICE_COUNTRY']} "
      f"(mean CATE {seg_country.iloc[0]['mean_cate']:.4f}, n={seg_country.iloc[0]['n']:.0f})")
print(f"Lowest-CATE country segment:  {seg_country.iloc[-1]['DEVICE_COUNTRY']} "
      f"(mean CATE {seg_country.iloc[-1]['mean_cate']:.4f}, n={seg_country.iloc[-1]['n']:.0f})")
print(f"Highest-CATE engagement tier: {seg_engagement.iloc[0]['ENGAGEMENT_TIER']} "
      f"(mean CATE {seg_engagement.iloc[0]['mean_cate']:.4f})")
print(f"Lowest-CATE engagement tier:  {seg_engagement.iloc[-1]['ENGAGEMENT_TIER']} "
      f"(mean CATE {seg_engagement.iloc[-1]['mean_cate']:.4f})")
