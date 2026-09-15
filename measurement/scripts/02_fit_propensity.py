"""
Step 2: Fit exposure-propensity model (EXPOSED ~ pre-period covariates).

We deliberately EXCLUDE any feature that is only observable conditional on
exposure/campaign delivery (CAMPAIGN_IMPRESSIONS) and any feature computed
from the campaign-flight period itself. All covariates here are pre-period
(as of PRE_CAMPAIGN_END = 2026-08-04) by construction of feature_query.sql.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_apple.parquet"
OUT_DATA_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"
MODEL_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_model.joblib"

df = pd.read_parquet(IN_PATH)
print("Loaded:", df.shape)

# ---------------------------------------------------------------
# Feature set: pre-period covariates only (no CAMPAIGN_IMPRESSIONS,
# no CONVERTED_POST_EXPOSURE -- those are post-treatment / outcome)
# ---------------------------------------------------------------
numeric_features = [
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE", "DAYS_SINCE_LAST_ACTIVE_MISSING",
    "HIST_IMPRESSIONS_7D", "HIST_IMP_EVENTS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_CTV_IMPRESSIONS_7D", "HIST_TILE_IMPRESSIONS_7D", "HIST_ACTIVE_DAYS_7D",
    "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS", "CHANNEL_TILE_RATIO",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "HIST_MOBILE_CAMPAIGN_RATIO", "HIST_WEB_CAMPAIGN_RATIO", "HIST_TV_CAMPAIGN_RATIO",
    "STV_YEAR", "DMA_CODE", "DEVICE_COUNTRY_MISSING",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
]
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY"]
# FIRMWARE_CODE / MODEL_CODE / DEVICE_LANGUAGE have very high cardinality;
# use one-hot only for the lower-cardinality, higher-signal categoricals here
# to keep the propensity model stable and interpretable.

# Bucket rare DEVICE_COUNTRY values into "OTHER": singleton categories (e.g.
# AE with 1 row, AR with 1 row) can perfectly separate EXPOSED for that one
# row, driving the corresponding one-hot coefficient toward +-infinity during
# logistic regression optimization -- this was the actual cause of the
# "divide by zero / overflow / invalid value in matmul" RuntimeWarnings seen
# during fitting (99.4% of rows are US or UNKNOWN; CA/MX/AE/AR together are
# <300 rows out of 5.74M and carry negligible signal on their own).
country_counts = df["DEVICE_COUNTRY"].value_counts()
rare_countries = country_counts[country_counts < 1000].index.tolist()
if rare_countries:
    print(f"Bucketing rare DEVICE_COUNTRY values into OTHER: {rare_countries}")
    df["DEVICE_COUNTRY"] = df["DEVICE_COUNTRY"].astype(str)
    df.loc[df["DEVICE_COUNTRY"].isin(rare_countries), "DEVICE_COUNTRY"] = "OTHER"

# Impute remaining numeric NaNs (STV_YEAR, DMA_CODE, DAYS_SINCE_LAST_ACTIVE)
# with median; missing-indicator flags already capture "missingness" as a
# signal for the model, so median imputation here just avoids NaN propagation.
for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    med = df[col].median()
    df[col] = df[col].fillna(med)

# Drop zero-variance numeric features: StandardScaler divides by std, and a
# constant column (std=0) produces NaN/inf during scaling, which silently
# corrupts the whole fit (surfaced as RuntimeWarning: divide by zero /
# overflow / invalid value in matmul during training). Confirmed
# HIST_CTV_IMPRESSIONS_7D is constant 0 across all 5.7M rows for this flight
# (Apple TV app has no CTV-tagged impressions in the pre-period window).
zero_var_cols = [c for c in numeric_features if df[c].std() == 0]
if zero_var_cols:
    print(f"Dropping zero-variance columns: {zero_var_cols}")
    numeric_features = [c for c in numeric_features if c not in zero_var_cols]

X_numeric = df[numeric_features].astype("float64")

# Log-transform heavy-tailed count/duration features before scaling.
# Diagnostic check (see /tmp/diagnose_nan.py investigation) found several
# raw count/duration columns had post-StandardScaler values of 100-500+
# standard deviations (e.g. HIST_CLICK_EVENTS_7D max ~539 sigma,
# TOTAL_VOD_TIME_7D max ~520 sigma) -- a small number of power-user rows
# with very high raw counts relative to the mean. Feeding these directly
# into gradient-based logistic regression after only z-scaling caused
# numerical overflow during optimization (RuntimeWarning: overflow /
# divide by zero / invalid value in matmul). log1p compresses the long
# tail while preserving rank order and is standard practice for
# count/duration features in propensity models.
log_transform_cols = [
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE",
    "HIST_IMPRESSIONS_7D", "HIST_IMP_EVENTS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_TILE_IMPRESSIONS_7D", "AVG_DAILY_IMPRESSIONS",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
]
log_transform_cols = [c for c in log_transform_cols if c in X_numeric.columns]
for col in log_transform_cols:
    # shift by the column min so log1p is always defined even if a rare
    # negative value slipped through (e.g. the single TOTAL_MINUTES=-131 row)
    shift = max(0, -X_numeric[col].min())
    X_numeric[col] = np.log1p(X_numeric[col] + shift)

X_categorical = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)

X = pd.concat([X_numeric, X_categorical], axis=1)
y = df["EXPOSED"].values

print("Feature matrix shape:", X.shape)

# Train/holdout split purely for AUC sanity-check of the propensity model's
# discriminative power (not used to filter the population -- we score everyone).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0, class_weight=None, solver="lbfgs")
model.fit(X_train_scaled, y_train)

train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# Refit on full data for final propensity scores (standard practice once
# AUC sanity-checked out-of-sample)
X_scaled_full = scaler.fit_transform(X)
model_full = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model_full.fit(X_scaled_full, y)

df["PROPENSITY_SCORE"] = model_full.predict_proba(X_scaled_full)[:, 1]

print("\nPropensity score distribution:")
print(df.groupby("EXPOSED")["PROPENSITY_SCORE"].describe())

# Feature importance (standardized coefficients)
coefs = pd.Series(model_full.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 15 |coefficient| features (standardized):")
print(coefs.head(15))

joblib.dump({"model": model_full, "scaler": scaler, "feature_cols": list(X.columns)}, MODEL_PATH)
df.to_parquet(OUT_DATA_PATH, index=False)
print(f"\nSaved scored data to {OUT_DATA_PATH}")
print(f"Saved model to {MODEL_PATH}")
