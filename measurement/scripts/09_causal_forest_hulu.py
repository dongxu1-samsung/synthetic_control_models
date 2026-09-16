"""
Step 9: Heterogeneous treatment effects (CATE) via Causal Forest + Double ML
(Hulu campaign 291260, flight 648768).

Why this step:
  - Every estimator so far (naive, matched, IPTW, AIPW) reports a single ATE
    number for the whole campaign. CausalForestDML estimates a *personalized*
    treatment effect tau(X) for every unit, which is directly actionable for
    targeting/budget allocation decisions in a way ATE alone isn't.
  - Uses EconML's CausalForestDML: an honest random forest for the CATE
    function, wrapped in Double Machine Learning (cross-fitted nuisance
    models for both the propensity/treatment model and the outcome model,
    same orthogonalization principle as the AIPW step). This is the most
    widely deployed CATE approach in industry (Microsoft, Uber, Netflix,
    Airbnb all use causal forests / GRF).

Compute note: CausalForestDML on the full 5.96M-row population is not
tractable in a reasonable wall-clock time with cv=3 nuisance cross-fitting +
an honest forest. We subsample to a stratified 750K-row working set (still
~2.5x the entire Apple TV treated population from the prior pipeline) --
statistically this is enormous overkill for a ~40-covariate CATE surface,
and the ATE recovered on this subsample is cross-checked against the full-
population AIPW ATE from 08_aipw_gbm_hulu.py as a validity check.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dml import CausalForestDML
import joblib

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_hulu.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/cate_scored_hulu.parquet"
MODEL_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/causal_forest_hulu.joblib"

RNG_SEED = 42
SUBSAMPLE_N = 750_000

df = pd.read_parquet(IN_PATH)
print("Full population:", df.shape)

# Stratified subsample by EXPOSED to preserve the ~12.3% treatment rate
df_sub, _ = train_test_split(
    df, train_size=SUBSAMPLE_N, stratify=df["EXPOSED"], random_state=RNG_SEED
)
df_sub = df_sub.reset_index(drop=True)
print(f"Working subsample: {df_sub.shape}, treated rate: {df_sub['EXPOSED'].mean():.4f} "
      f"(full population: {df['EXPOSED'].mean():.4f})")

numeric_features = [
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE", "DAYS_SINCE_LAST_ACTIVE_MISSING",
    "HIST_IMPRESSIONS_7D", "HIST_IMP_EVENTS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_ACTIVE_DAYS_7D",
    "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS", "CHANNEL_TILE_RATIO",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "HIST_MOBILE_CAMPAIGN_RATIO", "HIST_WEB_CAMPAIGN_RATIO", "HIST_TV_CAMPAIGN_RATIO",
    "STV_YEAR", "DMA_CODE", "DEVICE_COUNTRY_MISSING",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
]
# Low-cardinality categoricals one-hot encoded for the forest's split features
# (CausalForestDML's X argument doesn't take pandas "category" dtype the way
# LightGBM nuisance models do -- needs a fully numeric design matrix).
# High-cardinality categoricals (FIRMWARE_CODE 68 levels, MODEL_CODE 206
# levels, DEVICE_LANGUAGE 34 levels) are fed only to the LightGBM nuisance
# models (which handle them natively), not to the forest's X (would blow up
# the one-hot dimensionality and add little heterogeneity signal beyond what
# device/country/DMA already capture).
low_card_categoricals = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY"]
high_card_categoricals_for_nuisance_only = ["FIRMWARE_CODE", "DEVICE_LANGUAGE", "MODEL_CODE"]

country_counts = df_sub["DEVICE_COUNTRY"].value_counts()
rare_countries = country_counts[country_counts < 500].index.tolist()
df_sub["DEVICE_COUNTRY"] = df_sub["DEVICE_COUNTRY"].astype(str)
if rare_countries:
    df_sub.loc[df_sub["DEVICE_COUNTRY"].isin(rare_countries), "DEVICE_COUNTRY"] = "OTHER"

# Impute remaining numeric NaNs (STV_YEAR, DMA_CODE, DAYS_SINCE_LAST_ACTIVE)
# with median -- CausalForestDML's check_array call rejects NaN outright
# (sklearn's check_array under the hood), unlike LightGBM which handles NaN
# natively. Missing-indicator flags (DAYS_SINCE_LAST_ACTIVE_MISSING,
# DEVICE_COUNTRY_MISSING) are already in numeric_features to preserve the
# "missingness" signal for the model despite the imputation.
for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    med = df_sub[col].median()
    df_sub[col] = df_sub[col].fillna(med)

X_forest = pd.concat(
    [
        df_sub[numeric_features].astype("float64"),
        pd.get_dummies(df_sub[low_card_categoricals].astype(str), drop_first=True),
    ],
    axis=1,
)
forest_feature_names = list(X_forest.columns)
print(f"Forest CATE feature matrix (heterogeneity covariates): {X_forest.shape}")

# Nuisance-model feature set (used internally by CausalForestDML's model_y /
# model_t): include the high-cardinality categoricals too, since LightGBM
# handles them natively and richer nuisance models improve orthogonalization
# even if we don't want them as CATE-splitting features.
nuisance_categoricals = low_card_categoricals + high_card_categoricals_for_nuisance_only
for c in nuisance_categoricals:
    df_sub[c] = df_sub[c].astype(str).astype("category")
X_nuisance = pd.concat(
    [df_sub[numeric_features].astype("float64"), df_sub[nuisance_categoricals]],
    axis=1,
)

T = df_sub["EXPOSED"].values
Y = df_sub["CONVERTED_POST_EXPOSURE"].values

model_y = LGBMRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG_SEED, verbosity=-1,
)
model_t = LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG_SEED, verbosity=-1,
)

# NOTE: CausalForestDML fits model_y/model_t on (X, W) internally via its own
# cross-validation (cv=3 folds) -- it does NOT accept a categorical dtype
# directly for the W (nuisance-only) argument the way LightGBM's native API
# does when called standalone, so we one-hot the nuisance categoricals here
# too rather than relying on native categorical handling inside econml's
# wrapper.
W_nuisance = pd.concat(
    [
        df_sub[numeric_features].astype("float64"),
        pd.get_dummies(df_sub[nuisance_categoricals].astype(str), drop_first=True),
    ],
    axis=1,
)
print(f"Nuisance feature matrix (W, passed to model_y/model_t): {W_nuisance.shape}")

est = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,
    cv=3,
    n_estimators=400,
    min_samples_leaf=50,
    max_depth=None,
    honest=True,
    inference=True,
    n_jobs=-1,
    random_state=RNG_SEED,
)

print("\nFitting CausalForestDML (this may take several minutes)...")
est.fit(Y, T, X=X_forest.values, W=W_nuisance.values)
print("Fit complete.")

cate = est.effect(X_forest.values)
cate_lb, cate_ub = est.effect_interval(X_forest.values, alpha=0.05)

ate_forest = cate.mean()
print(f"\nMean CATE (== ATE recovered by the forest): {ate_forest:.5f} ({100*ate_forest:.4f} pp)")
print("Compare to:")
print("  - Matching (de-randomized):  0.03192 (91.5% rel. lift)")
print("  - IPTW (stabilized):          0.03325 (102.3% rel. lift)")
print("  - AIPW (LightGBM, full pop):  0.03156 (98.1% rel. lift)")

print(f"\nCATE distribution across the subsample:")
print(pd.Series(cate).describe())

df_sub["CATE"] = cate
df_sub["CATE_CI_LOWER"] = cate_lb
df_sub["CATE_CI_UPPER"] = cate_ub
df_sub.to_parquet(OUT_PATH, index=False)
joblib.dump({"model": est, "forest_feature_names": forest_feature_names}, MODEL_PATH)
print(f"\nSaved CATE-scored subsample to {OUT_PATH}")
print(f"Saved fitted CausalForestDML to {MODEL_PATH}")
