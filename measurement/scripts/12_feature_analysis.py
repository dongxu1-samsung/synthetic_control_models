"""
Feature analysis on the merged dataset: distributions, missingness,
correlation structure, and a first cut at identifying near-duplicate /
highly-correlated numeric features (beyond the exact-match dedup already
done in 11_merge_datasets.py).
"""
import numpy as np
import pandas as pd

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_features_apple.parquet"

df = pd.read_parquet(IN_PATH)
print("Shape:", df.shape)

label_cols = ["PSID", "EXPOSED", "CAMPAIGN_IMPRESSIONS", "CONVERTED_POST_EXPOSURE", "RAW_CONVERSIONS"]
feature_cols = [c for c in df.columns if c not in label_cols]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print(f"\n{len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features")
print("Categorical:", categorical_cols)

# ------------------------------------------------------------------
# Missingness
# ------------------------------------------------------------------
print("\n=== Missingness (features with >0 null) ===")
null_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
print(null_rates[null_rates > 0])

# ------------------------------------------------------------------
# Zero-variance check on the full merged set
# ------------------------------------------------------------------
print("\n=== Zero/near-zero variance numeric features ===")
for c in numeric_cols:
    std = df[c].std()
    if std == 0 or pd.isna(std):
        print(f"  {c}: std={std}, unique={df[c].nunique()}")

# ------------------------------------------------------------------
# High-cardinality categorical features
# ------------------------------------------------------------------
print("\n=== Categorical feature cardinality ===")
for c in categorical_cols:
    print(f"  {c}: {df[c].nunique()} distinct values")

# ------------------------------------------------------------------
# Correlation structure among numeric features -- flag highly correlated
# pairs (|corr| > 0.95) as candidate near-duplicates
# ------------------------------------------------------------------
print("\nComputing correlation matrix (numeric features only, this may take a moment)...")
corr = df[numeric_cols].corr()
corr_pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c = corr.iloc[i, j]
        if abs(c) > 0.90:
            corr_pairs.append((cols[i], cols[j], c))

corr_pairs_df = pd.DataFrame(corr_pairs, columns=["feature_a", "feature_b", "correlation"]).sort_values(
    "correlation", key=abs, ascending=False
)
print(f"\n=== Highly correlated feature pairs (|corr| > 0.90): {len(corr_pairs_df)} pairs ===")
pd.set_option("display.max_rows", None)
print(corr_pairs_df.to_string(index=False))

corr_pairs_df.to_csv(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/high_correlation_pairs.csv", index=False
)

# ------------------------------------------------------------------
# Descriptive stats for all numeric features
# ------------------------------------------------------------------
desc = df[numeric_cols].describe().T
desc.to_csv("/Users/dongd1.xu/Documents/hermes_agent/measurement/data/feature_descriptive_stats.csv")
print(f"\nSaved descriptive stats to feature_descriptive_stats.csv ({len(numeric_cols)} features)")
print("\nSaved high-correlation pairs to high_correlation_pairs.csv")
