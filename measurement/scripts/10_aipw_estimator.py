"""
Diagnostic #2: Augmented Inverse Propensity Weighting (AIPW / doubly robust
estimator).

AIPW combines an outcome regression model with the propensity weights. It
is consistent if EITHER the propensity model OR the outcome model is
correctly specified (hence "doubly robust"), and is the standard
recommended estimator when matching and IPTW disagree, since it uses both
sources of information and is less sensitive to misspecification of either.

Given the decile-stratified analysis (09_ps_stratified_lift.py) already
found strong evidence of genuine effect heterogeneity across the propensity
distribution (lift ranges from +90% in the lowest decile to -3% in the
highest), AIPW here serves to (a) provide a third point estimate for
cross-validation, and (b) produce a single ATE-style number that properly
accounts for the outcome model, rather than to "resolve" the divergence --
which the decile analysis already explained as heterogeneity rather than
bias in either existing method.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"

numeric_features = [
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE", "DAYS_SINCE_LAST_ACTIVE_MISSING",
    "HIST_IMPRESSIONS_7D", "HIST_IMP_EVENTS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_TILE_IMPRESSIONS_7D", "HIST_ACTIVE_DAYS_7D",
    "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS", "CHANNEL_TILE_RATIO",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
    "HIST_MOBILE_CAMPAIGN_RATIO", "HIST_WEB_CAMPAIGN_RATIO", "HIST_TV_CAMPAIGN_RATIO",
    "STV_YEAR", "DMA_CODE", "DEVICE_COUNTRY_MISSING",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
]
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY"]

df = pd.read_parquet(RAW_PATH)

# Common support trim (consistent with main pipeline)
p_exposed = df.loc[df.EXPOSED == 1, "PROPENSITY_SCORE"]
p_control = df.loc[df.EXPOSED == 0, "PROPENSITY_SCORE"]
lower = max(p_exposed.min(), p_control.min())
upper = min(p_exposed.max(), p_control.max())
df = df[(df.PROPENSITY_SCORE >= lower) & (df.PROPENSITY_SCORE <= upper)].copy()
print(f"Rows after common-support trim: {len(df)}")

for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    df[col] = df[col].fillna(df[col].median())

country_counts = df["DEVICE_COUNTRY"].value_counts()
rare_countries = country_counts[country_counts < 1000].index.tolist()
if rare_countries:
    df["DEVICE_COUNTRY"] = df["DEVICE_COUNTRY"].astype(str)
    df.loc[df["DEVICE_COUNTRY"].isin(rare_countries), "DEVICE_COUNTRY"] = "OTHER"

X_categorical = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)
X_all = pd.concat([df[numeric_features].astype("float64"), X_categorical], axis=1)
y_outcome = df["CONVERTED_POST_EXPOSURE"].values
treat = df["EXPOSED"].values
p_score = df["PROPENSITY_SCORE"].clip(1e-4, 1 - 1e-4).values

# ------------------------------------------------------------------
# Outcome model: predict P(converted | X) separately for treated and
# control arms, via 5-fold cross-fitting to avoid overfitting bias
# (standard practice for doubly-robust / AIPW estimators).
# Using HistGradientBoostingClassifier (handles nonlinearities/interactions
# natively, native NaN support) rather than logistic regression here, since
# the outcome model does not need to match the propensity model's
# functional form -- AIPW's robustness property only requires ONE of the
# two nuisance models to be correctly specified.
# ------------------------------------------------------------------
n = len(df)
mu1_hat = np.zeros(n)  # E[Y | X, T=1]
mu0_hat = np.zeros(n)  # E[Y | X, T=0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_arr = X_all.values

print("\nCross-fitting outcome models (5-fold)...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    t_train = treat[train_idx]
    y_train = y_outcome[train_idx]

    # Fit outcome model on treated subset of training fold
    treated_mask = t_train == 1
    control_mask = t_train == 0

    model_1 = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    model_1.fit(X_train[treated_mask], y_train[treated_mask])

    model_0 = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    model_0.fit(X_train[control_mask], y_train[control_mask])

    mu1_hat[test_idx] = model_1.predict_proba(X_test)[:, 1]
    mu0_hat[test_idx] = model_0.predict_proba(X_test)[:, 1]
    print(f"  Fold {fold+1}/5 done")

# ------------------------------------------------------------------
# AIPW estimator:
#   psi_i = mu1_hat_i - mu0_hat_i
#           + T_i * (Y_i - mu1_hat_i) / p_i
#           - (1-T_i) * (Y_i - mu0_hat_i) / (1 - p_i)
# ATE = mean(psi_i)
# ------------------------------------------------------------------
aipw_terms = (
    mu1_hat - mu0_hat
    + treat * (y_outcome - mu1_hat) / p_score
    - (1 - treat) * (y_outcome - mu0_hat) / (1 - p_score)
)
ate_aipw = aipw_terms.mean()
se_aipw = aipw_terms.std(ddof=1) / np.sqrt(n)
ci_lo = ate_aipw - 1.96 * se_aipw
ci_hi = ate_aipw + 1.96 * se_aipw

# For a relative-lift interpretation, use the AIPW-implied counterfactual
# control rate: mean(mu0_hat) adjusted by the control-arm correction term
control_counterfactual_rate = np.mean(
    mu0_hat + (1 - treat) * (y_outcome - mu0_hat) / (1 - p_score)
)
rel_lift_aipw = 100 * ate_aipw / control_counterfactual_rate

print("\n" + "=" * 80)
print("AIPW (doubly robust) RESULT")
print("=" * 80)
print(f"ATE (absolute lift): {ate_aipw:.5f}")
print(f"SE: {se_aipw:.5f}")
print(f"95% CI: [{ci_lo:.5f}, {ci_hi:.5f}]")
print(f"Implied counterfactual control rate: {control_counterfactual_rate:.5f}")
print(f"Relative lift: {rel_lift_aipw:.2f}%")

print("\n" + "=" * 80)
print("THREE-WAY COMPARISON")
print("=" * 80)
comparison = pd.DataFrame({
    "method": ["matched_1to1_caliper", "iptw_stabilized_trimmed", "aipw_doubly_robust"],
    "rel_lift_pct": [9.2091, 21.6098, rel_lift_aipw],
})
print(comparison.round(4).to_string(index=False))

comparison.to_csv("/Users/dongd1.xu/Documents/hermes_agent/measurement/data/three_way_comparison.csv", index=False)
print("\nSaved to /Users/dongd1.xu/Documents/hermes_agent/measurement/data/three_way_comparison.csv")
