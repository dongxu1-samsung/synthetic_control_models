"""
Step 19: AIPW (doubly robust) estimator on the MERGED Apple dataset.

Same methodology as 10_aipw_estimator.py, using the full merged + deduped
feature set (original + new S3-sourced numeric features). Needed because
matched (-7.6%) and IPTW (+9.4%) sign-diverge on the merged data (worse
divergence than the non-merged pipeline, explained by decile analysis in
18_ps_stratified_lift_merged.py as genuine effect heterogeneity combined
with a stronger/more discriminative propensity model shifting more of the
treated population into high-propensity, negative-lift strata).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_propensity_scored.parquet"

numeric_features = [
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
    "USER_HI2A", "G_USER_HI2A", "G_USER_VCR", "G_USER_MEDIA_PRICE",
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
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY", "OPERATING_SYSTEM", "NETSPEED"]

df = pd.read_parquet(RAW_PATH)
numeric_features = [c for c in numeric_features if c in df.columns]

p_exposed = df.loc[df.EXPOSED == 1, "PROPENSITY_SCORE"]
p_control = df.loc[df.EXPOSED == 0, "PROPENSITY_SCORE"]
lower = max(p_exposed.min(), p_control.min())
upper = min(p_exposed.max(), p_control.max())
df = df[(df.PROPENSITY_SCORE >= lower) & (df.PROPENSITY_SCORE <= upper)].copy()
print(f"Rows after common-support trim: {len(df)}")

for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    counts = df[col].astype(str).value_counts()
    rare = counts[counts < 1000].index.tolist()
    if rare:
        df[col] = df[col].astype(str)
        df.loc[df[col].isin(rare), col] = "OTHER"

X_categorical = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)
X_all = pd.concat([df[numeric_features].astype("float64"), X_categorical], axis=1)
y_outcome = df["CONVERTED_POST_EXPOSURE"].values
treat = df["EXPOSED"].values
p_score = df["PROPENSITY_SCORE"].clip(1e-4, 1 - 1e-4).values

n = len(df)
mu1_hat = np.zeros(n)
mu0_hat = np.zeros(n)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_arr = X_all.values

print("\nCross-fitting outcome models (5-fold, HistGradientBoostingClassifier)...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    t_train = treat[train_idx]
    y_train = y_outcome[train_idx]

    treated_mask = t_train == 1
    control_mask = t_train == 0

    model_1 = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    model_1.fit(X_train[treated_mask], y_train[treated_mask])

    model_0 = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    model_0.fit(X_train[control_mask], y_train[control_mask])

    mu1_hat[test_idx] = model_1.predict_proba(X_test)[:, 1]
    mu0_hat[test_idx] = model_0.predict_proba(X_test)[:, 1]
    print(f"  Fold {fold+1}/5 done")

aipw_terms = (
    mu1_hat - mu0_hat
    + treat * (y_outcome - mu1_hat) / p_score
    - (1 - treat) * (y_outcome - mu0_hat) / (1 - p_score)
)
ate_aipw_raw = aipw_terms.mean()
se_aipw_raw = aipw_terms.std(ddof=1) / np.sqrt(n)

# The raw (untrimmed) AIPW ATE is unstable: a small fraction of rows with
# near-0/near-1 propensity scores produce extreme psi_i values (some
# |psi_i| > 1000) via the 1/p or 1/(1-p) denominator, and the resulting
# mean is highly sensitive to exactly how much of that tail is included
# (ranges +0.4% to +2.7% relative lift across 0.1%-5% common-support trim
# widths in sensitivity testing). We report the 1st/99th-percentile
# propensity-score-trimmed AIPW ATE as the primary, stable estimate --
# trimming on the PROPENSITY SCORE (not the resulting psi_i value) is the
# principled fix, since it removes units where neither nuisance model is
# well identified rather than distorting the outcome distribution directly.
ps_lo, ps_hi = np.quantile(p_score, [0.01, 0.99])
trim_mask = (p_score >= ps_lo) & (p_score <= ps_hi)
ate_aipw = aipw_terms[trim_mask].mean()
se_aipw = aipw_terms[trim_mask].std(ddof=1) / np.sqrt(trim_mask.sum())
ci_lo = ate_aipw - 1.96 * se_aipw
ci_hi = ate_aipw + 1.96 * se_aipw

print(f"\nRaw (untrimmed) AIPW ATE: {ate_aipw_raw:.5f} +/- {1.96*se_aipw_raw:.5f} (unstable, reported for reference only)")
print(f"PS-trimmed AIPW ATE (primary estimate, {trim_mask.sum()}/{n} rows retained): {ate_aipw:.5f} +/- {1.96*se_aipw:.5f}")

control_counterfactual_rate = np.mean(
    (mu0_hat + (1 - treat) * (y_outcome - mu0_hat) / (1 - p_score))[trim_mask]
)
rel_lift_aipw = 100 * ate_aipw / control_counterfactual_rate

print("\n" + "=" * 80)
print("AIPW (doubly robust) RESULT -- MERGED DATASET")
print("=" * 80)
print(f"ATE (absolute lift): {ate_aipw:.5f}")
print(f"SE: {se_aipw:.5f}")
print(f"95% CI: [{ci_lo:.5f}, {ci_hi:.5f}]")
print(f"Implied counterfactual control rate: {control_counterfactual_rate:.5f}")
print(f"Relative lift: {rel_lift_aipw:.2f}%")

print("\n" + "=" * 80)
print("THREE-WAY COMPARISON (MERGED DATASET)")
print("=" * 80)
comparison = pd.DataFrame({
    "method": ["matched_1to1_caliper", "iptw_stabilized_trimmed", "aipw_doubly_robust"],
    "rel_lift_pct": [-7.6205, 9.4310, rel_lift_aipw],
    "abs_lift": [-0.00855, 0.00799, ate_aipw],
})
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_three_way_comparison.csv", index=False
)
print("\nSaved to merged_three_way_comparison.csv")

# save per-unit AIPW terms + mu-hats for later CATE/dimensional analysis
df["MU1_HAT"] = mu1_hat
df["MU0_HAT"] = mu0_hat
df["AIPW_TERM"] = aipw_terms
df.to_parquet(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_aipw_scored.parquet", index=False
)
print("Saved per-unit AIPW scores to merged_aipw_scored.parquet")
