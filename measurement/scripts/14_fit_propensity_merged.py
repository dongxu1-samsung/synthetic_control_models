"""
Fit exposure-propensity model on the MERGED dataset (our original features +
team-provided S3 features, deduplicated).

Same methodology as scripts/02_fit_propensity.py (log1p transform for
heavy-tailed count/duration features, drop zero-variance columns, bucket
rare categorical levels, exclude post-treatment/outcome fields), extended
to the new S3-sourced numeric features (ACR app usage, ad exposure,
day-part/language time-ratios) and additional categorical dimensions
(OPERATING_SYSTEM, NETSPEED, REGION, LANGUAGE).

High-cardinality categoricals (ISP: 3436 levels, MODEL_CODE: 171,
FIRMWARE_CODE: 33, REGION: 79 before bucketing) are excluded from the
propensity/outcome MODEL inputs (would need target encoding or tree-based
models to use safely) but are RETAINED in the dataset for the dimensional
CATE breakdown in a later step, where device/geo segments are exactly the
kind of cut we want to analyze.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_features_apple_deduped.parquet"
OUT_DATA_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_propensity_scored.parquet"
MODEL_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_propensity_model.joblib"

df = pd.read_parquet(IN_PATH)
print("Loaded:", df.shape)

# ---------------------------------------------------------------
# Numeric features: original set (from PROPENSITY_FEATURES_APPLE) plus
# new S3-sourced numeric features. Excludes labels (EXPOSED,
# CAMPAIGN_IMPRESSIONS, CONVERTED_POST_EXPOSURE, RAW_CONVERSIONS).
# ---------------------------------------------------------------
numeric_features = [
    # original features
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE", "DAYS_SINCE_LAST_ACTIVE_MISSING",
    "HIST_IMPRESSIONS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_ACTIVE_DAYS_7D",
    "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS", "CHANNEL_TILE_RATIO",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "HIST_MOBILE_CAMPAIGN_RATIO", "HIST_WEB_CAMPAIGN_RATIO", "HIST_TV_CAMPAIGN_RATIO",
    "STV_YEAR", "DMA_CODE",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
    # new S3-sourced numeric features
    "TAG_TYPE", "USER_HI2A", "G_USER_HI2A", "G_USER_VCR", "G_USER_MEDIA_PRICE",
    "G_USER_VIDEO_START_SUM", "G_USER_DISCREPANCY_PERCENT", "G_USER_IMPRESSION_21D",
    "ACR_APP_AVG_USING_TIME", "ACR_APP_USED_APP_CNT", "ACR_APP_EDUCATION",
    "ACR_APP_LIFESTYLE", "ACR_APP_INFORMATION", "ACR_APP_VIDEOS", "ACR_APP_SPORTS",
    "AD_COUNT", "AVG_AD_LENGTH", "AD_CATEGORY_COUNT",
    "AVG_DAILY_HOURS_WATCHED", "AVG_DAILY_APP_OPENS", "AVG_SESSION_TIME", "NUM_UNIQUE_APPS",
    "RATIO_TIME_INFORMATION", "RATIO_TIME_EDUCATION", "RATIO_TIME_GAME", "RATIO_TIME_VIDEOS",
    "RATIO_TIME_SPORTS", "RATIO_TIME_LIFESTYLE", "RATIO_TIME_AVOD_NULL",
    "RATIO_TIME_ENGLISH", "RATIO_TIME_FRENCH", "RATIO_TIME_SPANISH", "RATIO_TIME_GERMAN",
    "RATIO_TIME_POLISH", "RATIO_TIME_CHINESE",
]
numeric_features = [c for c in numeric_features if c in df.columns]

# Low-cardinality categoricals suitable for one-hot encoding in a linear model.
# High-cardinality dimensions (ISP, MODEL_CODE, FIRMWARE_CODE, and the
# unbucketed tail of REGION) are deliberately excluded from the MODEL here
# but retained in the full dataset for the dimensional CATE breakdown.
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY", "OPERATING_SYSTEM", "NETSPEED"]

for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    df[col] = df[col].fillna(df[col].median())

# Drop zero-variance numeric features (same defensive check as before --
# critical after StandardScaler failures found in the original pipeline)
zero_var_cols = [c for c in numeric_features if df[c].std() == 0]
if zero_var_cols:
    print(f"Dropping zero-variance columns: {zero_var_cols}")
    numeric_features = [c for c in numeric_features if c not in zero_var_cols]

# Bucket rare categorical levels (<1000 rows) to avoid quasi-separation,
# same fix applied to DEVICE_COUNTRY in the original pipeline
for col in categorical_features:
    counts = df[col].astype(str).value_counts()
    rare = counts[counts < 1000].index.tolist()
    if rare:
        print(f"Bucketing rare {col} values into OTHER: {len(rare)} levels, "
              f"{counts[rare].sum()} rows")
        df[col] = df[col].astype(str)
        df.loc[df[col].isin(rare), col] = "OTHER"

X_numeric = df[numeric_features].astype("float64")

# Log-transform heavy-tailed count/duration features before scaling (same
# fix as the original pipeline -- confirmed necessary again here since the
# new S3 features include similarly skewed ad-exposure/usage counts)
log_transform_cols = [
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE",
    "HIST_IMPRESSIONS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "AVG_DAILY_IMPRESSIONS",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
    "G_USER_VIDEO_START_SUM", "G_USER_IMPRESSION_21D",
    "ACR_APP_AVG_USING_TIME", "ACR_APP_USED_APP_CNT", "ACR_APP_EDUCATION",
    "ACR_APP_LIFESTYLE", "ACR_APP_INFORMATION", "ACR_APP_VIDEOS", "ACR_APP_SPORTS",
    "AD_COUNT", "AVG_AD_LENGTH", "AD_CATEGORY_COUNT",
    "AVG_DAILY_HOURS_WATCHED", "AVG_DAILY_APP_OPENS", "AVG_SESSION_TIME", "NUM_UNIQUE_APPS",
]
log_transform_cols = [c for c in log_transform_cols if c in X_numeric.columns]
for col in log_transform_cols:
    shift = max(0, -X_numeric[col].min())
    X_numeric[col] = np.log1p(X_numeric[col] + shift)

X_categorical = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)

X = pd.concat([X_numeric, X_categorical], axis=1)
y = df["EXPOSED"].values

print("Feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model.fit(X_train_scaled, y_train)

train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

X_scaled_full = scaler.fit_transform(X)
model_full = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model_full.fit(X_scaled_full, y)

df["PROPENSITY_SCORE"] = model_full.predict_proba(X_scaled_full)[:, 1]

print("\nPropensity score distribution:")
print(df.groupby("EXPOSED")["PROPENSITY_SCORE"].describe())

coefs = pd.Series(model_full.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 20 |coefficient| features (standardized):")
print(coefs.head(20))

joblib.dump(
    {"model": model_full, "scaler": scaler, "feature_cols": list(X.columns),
     "train_auc": train_auc, "test_auc": test_auc},
    MODEL_PATH,
)
df.to_parquet(OUT_DATA_PATH, index=False)
print(f"\nSaved scored data to {OUT_DATA_PATH}")
print(f"Saved model to {MODEL_PATH}")

coefs.to_csv("/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_propensity_coefficients.csv")
