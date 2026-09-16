"""
Step 8b: Placebo test for the AIPW/doubly-robust estimator (Hulu campaign
291260, flight 648768).

Same logic as 06c_placebo_iptw_hulu.py: split the CONTROL pool into two
pseudo-groups (both truly unexposed), refit nuisance models (propensity +
outcome) on the pseudo-treatment label, and compute the AIPW pseudo-outcome.
If the pipeline is unbiased, the measured placebo "lift" should be ~0.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_hulu.parquet"
rng = np.random.default_rng(42)
N_FOLDS = 5
N_BOOTSTRAP = 1000

raw = pd.read_parquet(RAW_PATH)
control_pool = raw[raw.EXPOSED == 0].copy().reset_index(drop=True)

shuffled_idx = rng.permutation(len(control_pool))
half = len(control_pool) // 2
pseudo_treat = np.zeros(len(control_pool), dtype=int)
pseudo_treat[shuffled_idx[:half]] = 1
control_pool["PSEUDO_TREAT"] = pseudo_treat

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
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY", "FIRMWARE_CODE", "DEVICE_LANGUAGE", "MODEL_CODE"]
feature_cols = numeric_features + categorical_features

X = control_pool[feature_cols].copy()
for c in categorical_features:
    X[c] = X[c].astype(str).astype("category")
for c in numeric_features:
    X[c] = X[c].astype("float64")

T = control_pool["PSEUDO_TREAT"].values
Y = control_pool["CONVERTED_POST_EXPOSURE"].values
n = len(control_pool)
print(f"Placebo pool: {n} rows, pseudo-treated: {T.sum()}, pseudo-control: {(1-T).sum()}")

propensity_oof = np.zeros(n)
mu1_oof = np.zeros(n)
mu0_oof = np.zeros(n)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
lgbm_params_clf = dict(n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
lgbm_params_reg = dict(n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=63,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, T)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    T_train = T[train_idx]
    Y_train = Y[train_idx]

    clf = LGBMClassifier(**lgbm_params_clf)
    clf.fit(X_train, T_train, categorical_feature=categorical_features)
    propensity_oof[test_idx] = clf.predict_proba(X_test)[:, 1]

    train_treated_mask = T_train == 1
    train_control_mask = T_train == 0
    reg1 = LGBMRegressor(**lgbm_params_reg)
    reg1.fit(X_train[train_treated_mask], Y_train[train_treated_mask], categorical_feature=categorical_features)
    mu1_oof[test_idx] = reg1.predict(X_test)
    reg0 = LGBMRegressor(**lgbm_params_reg)
    reg0.fit(X_train[train_control_mask], Y_train[train_control_mask], categorical_feature=categorical_features)
    mu0_oof[test_idx] = reg0.predict(X_test)

    print(f"Fold {fold+1}/{N_FOLDS} done")

auc = roc_auc_score(T, propensity_oof)
print(f"\nPseudo-propensity AUC (should be ~0.5, assignment was random): {auc:.4f}")

mu1_oof = np.clip(mu1_oof, 0, 1)
mu0_oof = np.clip(mu0_oof, 0, 1)
e = np.clip(propensity_oof, 1e-3, 1 - 1e-3)

psi = mu1_oof - mu0_oof + T * (Y - mu1_oof) / e - (1 - T) * (Y - mu0_oof) / (1 - e)
placebo_ate = psi.mean()

boot = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, n)
    boot[b] = psi[idx].mean()
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())

print("\n--- Placebo test (AIPW, LightGBM nuisance models refit on pseudo-split) ---")
print(f"Placebo AIPW ATE (should be ~0): {placebo_ate:.5f}")
print(f"95% CI: [{ci_lo:.5f}, {ci_hi:.5f}], bootstrap p={p_boot:.4f}")
