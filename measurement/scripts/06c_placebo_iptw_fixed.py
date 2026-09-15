"""
Step 6b (corrected): Placebo test for IPTW.

The original placebo script incorrectly reused the REAL propensity score
(fit to predict actual EXPOSED) against a random 50/50 pseudo-split. Since
pseudo-assignment is independent of the real propensity score, the
stabilized-weight formulas pull in opposite directions on the real score and
manufacture an artificial "lift" between two statistically identical halves.

Fix: refit a propensity model on the SAME covariates but against the
pseudo-treatment label (which is random by construction), then apply IPTW
using those refit scores. If the modeling+weighting pipeline is unbiased,
the measured placebo lift should be ~0 regardless of which covariates went
into the (here, uninformative-by-construction) propensity model.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"
rng = np.random.default_rng(42)

raw = pd.read_parquet(RAW_PATH)
control_pool = raw[raw.EXPOSED == 0].copy().reset_index(drop=True)

shuffled_idx = rng.permutation(len(control_pool))
half = len(control_pool) // 2
pseudo_idx_treated = shuffled_idx[:half]
pseudo_idx_control = shuffled_idx[half:]

pseudo_df = control_pool.copy()
pseudo_df["PSEUDO_TREAT"] = 0
pseudo_df.loc[pseudo_idx_treated, "PSEUDO_TREAT"] = 1

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
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY"]

country_counts = pseudo_df["DEVICE_COUNTRY"].value_counts()
rare_countries = country_counts[country_counts < 1000].index.tolist()
if rare_countries:
    pseudo_df["DEVICE_COUNTRY"] = pseudo_df["DEVICE_COUNTRY"].astype(str)
    pseudo_df.loc[pseudo_df["DEVICE_COUNTRY"].isin(rare_countries), "DEVICE_COUNTRY"] = "OTHER"

for col in ["STV_YEAR", "DMA_CODE", "DAYS_SINCE_LAST_ACTIVE"]:
    pseudo_df[col] = pseudo_df[col].fillna(pseudo_df[col].median())

X_numeric = pseudo_df[numeric_features].astype("float64")
for col in log_transform_cols:
    shift = max(0, -X_numeric[col].min())
    X_numeric[col] = np.log1p(X_numeric[col] + shift)
X_categorical = pd.get_dummies(pseudo_df[categorical_features].astype(str), drop_first=True)
X = pd.concat([X_numeric, X_categorical], axis=1)
y = pseudo_df["PSEUDO_TREAT"].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model.fit(Xs, y)
pseudo_df["PSEUDO_PROPENSITY"] = model.predict_proba(Xs)[:, 1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, pseudo_df["PSEUDO_PROPENSITY"])
print(f"Pseudo-propensity model AUC (should be ~0.5, since assignment was random): {auc:.4f}")

# Stabilized IPTW using the pseudo-propensity
p = pseudo_df["PSEUDO_PROPENSITY"].clip(1e-4, 1 - 1e-4)
p_marginal = y.mean()
sw = np.where(y == 1, p_marginal / p, (1 - p_marginal) / (1 - p))
lo, hi = np.percentile(sw, [1, 99])
sw_trim = np.clip(sw, lo, hi)

conv = pseudo_df["CONVERTED_POST_EXPOSURE"].values
rate_t = np.average(conv[y == 1], weights=sw_trim[y == 1])
rate_c = np.average(conv[y == 0], weights=sw_trim[y == 0])
placebo_iptw_lift = rate_t - rate_c

t_vals = conv[y == 1]
c_vals = conv[y == 0]
w_t = sw_trim[y == 1]
w_c = sw_trim[y == 0]
n_t, n_c = len(t_vals), len(c_vals)
N_BOOT = 1000
boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx_t = rng.integers(0, n_t, n_t)
    idx_c = rng.integers(0, n_c, n_c)
    r_t = np.average(t_vals[idx_t], weights=w_t[idx_t])
    r_c = np.average(c_vals[idx_c], weights=w_c[idx_c])
    boot[b] = r_t - r_c
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())

print("\n--- Placebo test (IPTW, corrected -- refit propensity on pseudo-split) ---")
print(f"Pseudo-treated weighted conv rate: {rate_t:.5f}")
print(f"Pseudo-control weighted conv rate: {rate_c:.5f}")
print(f"Placebo IPTW lift (should be ~0): {placebo_iptw_lift:.5f}")
print(f"95% CI: [{ci_lo:.5f}, {ci_hi:.5f}], bootstrap p={p_boot:.4f}")
