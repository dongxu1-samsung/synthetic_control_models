"""
Step 8: Doubly-robust (AIPW) ATE estimation with gradient-boosted (LightGBM)
nuisance models, cross-fitted (Hulu campaign 291260, flight 648768).

Why this step:
  - Matching and IPTW already broadly agree (91.5% vs 102.3% relative lift),
    but both depend on the SAME propensity model (logistic regression,
    AUC ~0.622). AIPW/doubly-robust estimation is consistent if EITHER the
    propensity model OR the outcome model is correctly specified -- a
    strictly more robust check than either matching or IPTW alone, and it
    lets us swap in a more flexible nuisance model (LightGBM) to see if a
    weak linear propensity model was leaving lift on the table.
  - Cross-fitting (K-fold, out-of-fold nuisance predictions) avoids the
    overfitting bias that would result from evaluating a flexible ML model
    on the same rows it was trained on (standard practice in the Double
    Machine Learning / AIPW literature -- Chernozhukov et al. 2018).

AIPW pseudo-outcome (per unit i):
  psi_i = mu1_i - mu0_i
          + T_i     * (Y_i - mu1_i) / e_i
          - (1-T_i) * (Y_i - mu0_i) / (1 - e_i)
ATE = mean(psi_i); doubly-robust because it stays consistent if either
e_i (propensity) or mu_i (outcome) is correctly specified.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_hulu.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/aipw_scored_hulu.parquet"

N_FOLDS = 5
RNG_SEED = 42
N_BOOTSTRAP = 2000
rng = np.random.default_rng(RNG_SEED)

df = pd.read_parquet(IN_PATH)
print("Loaded:", df.shape)

# ------------------------------------------------------------------
# Feature set: same pre-period covariates as the propensity model (02_*),
# but LightGBM handles high-cardinality categoricals natively (as
# integer-encoded "category" dtype), so we use FIRMWARE_CODE, DEVICE_LANGUAGE,
# and MODEL_CODE here too instead of dropping them for cardinality reasons
# (the original logistic-regression pipeline dropped them to keep one-hot
# encoding tractable -- LightGBM's native categorical splitting sidesteps
# that limitation and may recover extra signal).
# ------------------------------------------------------------------
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
# HIST_CTV_IMPRESSIONS_7D and HIST_TILE_IMPRESSIONS_7D dropped: constant-zero
# and perfectly-collinear-with-HIST_IMP_EVENTS_7D respectively for this
# population (same issues found in 02_fit_propensity_hulu.py). Tree models
# don't strictly need either fix (GBM splitting handles constant/collinear
# columns gracefully), but we drop them anyway for consistency with the
# other nuisance model and to keep apples-to-apples covariate sets.
categorical_features = ["DEVICE_COUNTRY", "SAMSUNG_AFFINITY", "FIRMWARE_CODE", "DEVICE_LANGUAGE", "MODEL_CODE"]

feature_cols = numeric_features + categorical_features

X = df[feature_cols].copy()
for c in categorical_features:
    X[c] = X[c].astype(str).astype("category")
for c in numeric_features:
    X[c] = X[c].astype("float64")

T = df["EXPOSED"].values
Y = df["CONVERTED_POST_EXPOSURE"].values
n = len(df)
print(f"Feature matrix: {X.shape}, treated: {T.sum()}, control: {(1-T).sum()}")

# ------------------------------------------------------------------
# Cross-fitted nuisance models
# ------------------------------------------------------------------
propensity_oof = np.zeros(n)
mu1_oof = np.zeros(n)  # E[Y | X, T=1], predicted for every unit (factual + counterfactual)
mu0_oof = np.zeros(n)  # E[Y | X, T=0], predicted for every unit

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

lgbm_params_clf = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG_SEED, verbosity=-1,
)
lgbm_params_reg = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG_SEED, verbosity=-1,
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, T)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    T_train = T[train_idx]
    Y_train = Y[train_idx]

    # --- propensity model: P(T=1 | X), trained on this fold's train split ---
    clf = LGBMClassifier(**lgbm_params_clf)
    clf.fit(X_train, T_train, categorical_feature=categorical_features)
    propensity_oof[test_idx] = clf.predict_proba(X_test)[:, 1]

    # --- outcome models: separate regressions for treated / control rows
    # within this fold's train split (T-learner style nuisance), predicted
    # out-of-fold for ALL held-out rows regardless of their actual treatment
    # (we need both mu1 and mu0 for every unit -- one is factual, one is the
    # counterfactual prediction used by the AIPW correction term). ---
    train_treated_mask = T_train == 1
    train_control_mask = T_train == 0

    reg1 = LGBMRegressor(**lgbm_params_reg)
    reg1.fit(X_train[train_treated_mask], Y_train[train_treated_mask],
             categorical_feature=categorical_features)
    mu1_oof[test_idx] = reg1.predict(X_test)

    reg0 = LGBMRegressor(**lgbm_params_reg)
    reg0.fit(X_train[train_control_mask], Y_train[train_control_mask],
             categorical_feature=categorical_features)
    mu0_oof[test_idx] = reg0.predict(X_test)

    fold_auc = roc_auc_score(T[test_idx], propensity_oof[test_idx])
    print(f"Fold {fold+1}/{N_FOLDS}: propensity AUC (out-of-fold) = {fold_auc:.4f}, "
          f"mu1 range=[{mu1_oof[test_idx].min():.4f}, {mu1_oof[test_idx].max():.4f}], "
          f"mu0 range=[{mu0_oof[test_idx].min():.4f}, {mu0_oof[test_idx].max():.4f}]")

overall_auc = roc_auc_score(T, propensity_oof)
print(f"\nOverall out-of-fold propensity AUC (LightGBM, cross-fitted): {overall_auc:.4f}")
print("(compare to logistic regression's 0.622 from 02_fit_propensity_hulu.py)")

# Clip mu predictions to [0, 1] (they're probabilities of a binary outcome;
# LGBMRegressor with default MSE loss can slightly overshoot near 0/1)
mu1_oof = np.clip(mu1_oof, 0, 1)
mu0_oof = np.clip(mu0_oof, 0, 1)

# ------------------------------------------------------------------
# AIPW pseudo-outcome and ATE
# ------------------------------------------------------------------
e = np.clip(propensity_oof, 1e-3, 1 - 1e-3)  # trim to avoid extreme weights (same 1e-3 style as IPTW step)

psi = (
    mu1_oof - mu0_oof
    + T * (Y - mu1_oof) / e
    - (1 - T) * (Y - mu0_oof) / (1 - e)
)

ate_aipw = psi.mean()
se_analytic = psi.std(ddof=1) / np.sqrt(n)

# Bootstrap CI (resample psi directly -- valid since AIPW scores are
# asymptotically i.i.d. once nuisance functions are fixed via cross-fitting)
boot_ates = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, n)
    boot_ates[b] = psi[idx].mean()
ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])
p_boot = 2 * min((boot_ates <= 0).mean(), (boot_ates >= 0).mean())

rate_c_baseline = Y[T == 0].mean()  # for a relative-lift denominator comparable to other methods
rel_lift_pct = 100 * ate_aipw / rate_c_baseline

print("\n" + "=" * 90)
print("AIPW (DOUBLY-ROBUST) ATE ESTIMATE -- Hulu Flight 648768 (Campaign 291260)")
print("=" * 90)
print(f"ATE (AIPW):            {ate_aipw:.5f}  ({100*ate_aipw:.4f} pp)")
print(f"Analytic SE:            {se_analytic:.5f}")
print(f"Analytic 95% CI:        [{ate_aipw - 1.96*se_analytic:.5f}, {ate_aipw + 1.96*se_analytic:.5f}]")
print(f"Bootstrap 95% CI:        [{ci_lo:.5f}, {ci_hi:.5f}]")
print(f"Bootstrap p-value:       {p_boot:.5f}")
print(f"Rel. lift vs. raw control conv rate ({rate_c_baseline:.4f}): {rel_lift_pct:.2f}%")

# Sanity check: compare cross-fitted mu1/mu0 to naive/matched/IPTW group means
print(f"\nMean predicted mu1 (E[Y|X,T=1]) among treated: {mu1_oof[T==1].mean():.5f} "
      f"(actual treated conv rate: {Y[T==1].mean():.5f})")
print(f"Mean predicted mu0 (E[Y|X,T=0]) among control: {mu0_oof[T==0].mean():.5f} "
      f"(actual control conv rate: {Y[T==0].mean():.5f})")

df["PROPENSITY_SCORE_LGBM"] = propensity_oof
df["MU1_LGBM"] = mu1_oof
df["MU0_LGBM"] = mu0_oof
df["AIPW_PSI"] = psi
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved AIPW-scored data to {OUT_PATH}")
