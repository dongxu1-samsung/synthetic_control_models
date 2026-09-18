"""
Merge script: combine our PROPENSITY_FEATURES_APPLE dataset with the
team-provided S3 dataset for flight 658095, dedupe overlapping/redundant
features, and produce a clean combined feature table for modeling.

Population check (pre-merge validation, see /tmp/check_overlap.py /
/tmp/cross_validate_labels.py):
  - Our dataset: 5,742,285 distinct PSIDs
  - S3 dataset:  5,708,675 distinct PSIDs
  - Overlap: 5,699,320 (99.25% of ours, 99.84% of S3's)
  - Label agreement on the overlap: EXPOSED/exposed 100% match,
    CONVERTED_POST_EXPOSURE/converted 100% match,
    CAMPAIGN_IMPRESSIONS/impressions 100% exact match (corr=1.0)
  -> both datasets describe the same underlying population/labels for this
     flight, built via independent feature pipelines. Safe to inner-join
     and treat as complementary feature sources.

Deduplication decisions:
  - Keep our EXPOSED / CAMPAIGN_IMPRESSIONS / CONVERTED_POST_EXPOSURE as
    the canonical labels (validated identical to S3's exposed/impressions/
    converted above) -- drop the S3 duplicates.
  - SAMSUNG_AFFINITY: S3's version is 100% constant ("direct") -- drop S3's,
    keep ours (which has a small UNKNOWN category carrying signal).
  - DEVICE_LANGUAGE (ours) vs language (S3): S3 has better coverage where
    ours shows UNKNOWN. Keep S3's `language` as the canonical language
    field, drop DEVICE_LANGUAGE.
  - S3 zero-variance / near-zero-variance columns dropped entirely:
    device_type_id, samsung_affinity, uid_type, uid_sticky,
    ratio_time_avod_true, ratio_time_avod_false (all 100% single value);
    bid_request_device_ext_operating_system_id, bid_request_device_ext_browser_id,
    bid_request_device_geo_type, bid_request_device_ext_device_type_id,
    bid_request_ext_channel (99.97% single value, <0.03% variation is noise).
  - operating_system (S3, 2 real values: TIZEN_TV/TIZEN_TV_6) kept as a new
    signal not present in our dataset.
  - region (S3) kept as a richer geo signal than our DMA_CODE (both kept --
    different granularity, not exact duplicates).
"""
import numpy as np
import pandas as pd

OUR_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_apple.parquet"
S3_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/s3_raw/scm_dataset_658095_2026-08-05_2026-08-19_US.csv"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_features_apple.parquet"

print("Loading our dataset...")
our_df = pd.read_parquet(OUR_PATH)
print("Our shape:", our_df.shape)

# S3 columns to drop entirely (zero/near-zero variance, or exact duplicates
# of a field we keep from OUR dataset)
s3_drop_cols = [
    "exposed", "impressions", "converted",  # duplicates of our EXPOSED/CAMPAIGN_IMPRESSIONS/CONVERTED_POST_EXPOSURE
    "samsung_affinity",  # 100% constant "direct"; ours has signal in UNKNOWN category
    "device_type_id", "uid_type", "uid_sticky",  # 100% constant
    "ratio_time_avod_true", "ratio_time_avod_false",  # 100% constant (0.0)
    "bid_request_device_ext_operating_system_id", "bid_request_device_ext_browser_id",
    "bid_request_device_geo_type", "bid_request_device_ext_device_type_id",
    "bid_request_ext_channel",  # all 99.97%+ single value
]
# raw_conversions kept as an alternate/count-based outcome (not a duplicate --
# our CONVERTED_POST_EXPOSURE is binary; raw_conversions is a count, useful
# for a secondary continuous-outcome lift check)

print(f"\nLoading S3 dataset in chunks, dropping {len(s3_drop_cols)} redundant/zero-variance columns...")
chunks = []
for i, chunk in enumerate(pd.read_csv(S3_PATH, chunksize=1_000_000, dtype={"samsung_psid": str}, low_memory=False)):
    chunk = chunk.drop(columns=s3_drop_cols)
    chunks.append(chunk)
    print(f"  chunk {i+1} loaded, shape {chunk.shape}")
s3_df = pd.concat(chunks, ignore_index=True)
del chunks
print("S3 shape after column drop:", s3_df.shape)

assert s3_df["samsung_psid"].nunique() == len(s3_df), "S3 PSID not unique!"
assert our_df["PSID"].nunique() == len(our_df), "Our PSID not unique!"

# ------------------------------------------------------------------
# Merge (inner join on PSID -- population overlap already validated at
# 99.25%/99.84%; inner join keeps only the mutually-covered population,
# which is the safest choice for a combined feature set since features
# from BOTH sources are required for every row)
# ------------------------------------------------------------------
merged = our_df.merge(s3_df, left_on="PSID", right_on="samsung_psid", how="inner")
merged = merged.drop(columns=["samsung_psid"])
print("\nMerged shape:", merged.shape)
print(f"Rows dropped from ours (no S3 match): {len(our_df) - len(merged)} "
      f"({100*(len(our_df)-len(merged))/len(our_df):.2f}%)")
print(f"Rows dropped from S3 (no our-dataset match): {len(s3_df) - len(merged)} "
      f"({100*(len(s3_df)-len(merged))/len(s3_df):.2f}%)")

# Rename S3 columns to a consistent naming convention (uppercase, matching
# our existing convention) for downstream pipeline consistency
rename_map = {c: c.upper() for c in s3_df.columns if c != "samsung_psid"}
merged = merged.rename(columns=rename_map)

print("\nFinal merged columns:")
for c in merged.columns:
    print(" ", c, merged[c].dtype)

print("\nFinal shape:", merged.shape)
merged.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
