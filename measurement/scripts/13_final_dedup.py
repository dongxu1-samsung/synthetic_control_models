"""
Final feature dedup based on correlation analysis (12_feature_analysis.py),
producing the clean feature set used for modeling.

Dedup decisions (threshold: |corr| > 0.98 -> treat as duplicate, keep one):
  - HIST_IMPRESSIONS_7D / HIST_IMP_EVENTS_7D / HIST_TILE_IMPRESSIONS_7D
    (corr ~1.0 pairwise) -> keep HIST_IMPRESSIONS_7D only
  - G_USER_MEDIA_PRICE / G_USER_ADJUSTED_MEDIA_PRICE / G_USER_BUYER_PRICE
    (corr 0.999+) -> keep G_USER_MEDIA_PRICE only
  - G_USER_VIDEO_COMPLETE_SUM / G_USER_VIDEO_START_SUM (corr 0.999) ->
    keep G_USER_VIDEO_START_SUM only
  - AD_COUNT / UNIQUE_ADS_COUNT / TOTAL_AD_LENGTH (corr 0.99+) -> keep
    AD_COUNT only (BRAND_COUNT / ADVERTISER_COUNT / AD_CATEGORY_COUNT kept
    separately -- correlated with AD_COUNT at 0.94-0.98 but capture a
    distinct dimension: count of unique advertisers/brands/categories
    rather than raw ad volume)
  - CHANNEL_TILE_RATIO / DEVICE_COUNTRY_MISSING (corr = -1.0 exactly,
    structurally coincidental -- both derived from whether the user had
    any pre-period impression row) -> keep CHANNEL_TILE_RATIO (continuous,
    more informative), drop DEVICE_COUNTRY_MISSING

Also drops HIST_CTV_IMPRESSIONS_7D (zero variance, confirmed in earlier
pipeline -- Apple TV app has no CTV-tagged impressions in this pre-period).
"""
import pandas as pd

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_features_apple.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_features_apple_deduped.parquet"

df = pd.read_parquet(IN_PATH)
print("Shape before final dedup:", df.shape)

drop_cols = [
    "HIST_IMP_EVENTS_7D", "HIST_TILE_IMPRESSIONS_7D",  # keep HIST_IMPRESSIONS_7D
    "G_USER_ADJUSTED_MEDIA_PRICE", "G_USER_BUYER_PRICE",  # keep G_USER_MEDIA_PRICE
    "G_USER_VIDEO_COMPLETE_SUM",  # keep G_USER_VIDEO_START_SUM
    "UNIQUE_ADS_COUNT", "TOTAL_AD_LENGTH",  # keep AD_COUNT
    "ADVERTISER_COUNT", "BRAND_COUNT",  # keep AD_COUNT -- corrected decision:
    # log1p(AD_COUNT)/log1p(ADVERTISER_COUNT)/log1p(BRAND_COUNT) correlate at
    # 0.995-0.999 (condition number ~70 for the 3-feature block), i.e. these
    # are effectively duplicates once log-transformed for modeling, not
    # distinct dimensions as originally assumed from raw correlations
    # (0.94-0.98). Confirmed empirically: fitting all three produced unstable,
    # oversized opposite-signed coefficients (BRAND_COUNT -2.09, AD_COUNT
    # +1.70) -- a classic multicollinearity symptom. Keeping only AD_COUNT.
    "DEVICE_COUNTRY_MISSING",  # keep CHANNEL_TILE_RATIO
    "HIST_CTV_IMPRESSIONS_7D",  # zero variance
]
drop_cols = [c for c in drop_cols if c in df.columns]
print(f"\nDropping {len(drop_cols)} redundant/zero-variance columns: {drop_cols}")

df = df.drop(columns=drop_cols)
print("Shape after final dedup:", df.shape)

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")

label_cols = ["PSID", "EXPOSED", "CAMPAIGN_IMPRESSIONS", "CONVERTED_POST_EXPOSURE", "RAW_CONVERSIONS"]
feature_cols = [c for c in df.columns if c not in label_cols]
print(f"\nFinal feature count: {len(feature_cols)}")
print(feature_cols)
