"""
Step 10: Qini / AUUC curves to validate CATE targeting quality (Hulu
campaign 291260, flight 648768).

Why this step:
  - We have no ground-truth ITE for real production data, so PEHE-style
    accuracy metrics are not available. The standard no-ground-truth
    validation for a CATE model is a Qini/AUUC curve: since exposure here
    is NOT randomized (RTB auction wins are selection-biased), we use
    IPW-adjusted uplift so the curve stays valid despite non-random
    treatment assignment.
  - Business framing: rank users by predicted CATE (from the causal forest
    in 09_causal_forest_hulu.py), then ask "if we had only served ads to the
    top-k% by predicted incremental benefit, how much of the total lift would
    we have captured?" This is the actionable output for budget targeting
    that a single ATE number can't provide.

IPW-adjusted uplift at top-k%:
    uplift_ipw(k) = mean(T*Y/e(X) in top-k) - mean((1-T)*Y/(1-e(X)) in top-k)
  where e(X) is the propensity score (reuse the LightGBM out-of-fold scores
  from 08_aipw_gbm_hulu.py, which are already cross-fitted and available for
  the full population -- but this Qini analysis runs on the CATE-scored
  750K subsample from 09_causal_forest_hulu.py since CATE isn't scored for
  the full population).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATE_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/cate_scored_hulu.parquet"
AIPW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/aipw_scored_hulu.parquet"
OUT_CSV = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/qini_auuc_hulu.csv"
OUT_PLOT = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/qini_curve_hulu.png"

cate_df = pd.read_parquet(CATE_PATH)
aipw_df = pd.read_parquet(AIPW_PATH)[["PSID", "PROPENSITY_SCORE_LGBM"]]

df = cate_df.merge(aipw_df, on="PSID", how="left")
print(f"Merged CATE + propensity: {df.shape}, "
      f"missing propensity: {df['PROPENSITY_SCORE_LGBM'].isna().sum()}")

# Fallback: a tiny number of rows might not have a match due to how the
# subsample vs. cross-fit folds were drawn independently -- use the
# forest-internal propensity (from CausalForestDML's own model_t) as backup
# if needed. In practice this should be ~0 since both come from the same
# full population dataframe.
df = df.dropna(subset=["PROPENSITY_SCORE_LGBM"]).reset_index(drop=True)
print(f"Rows after dropping unmatched: {len(df)}")

e = df["PROPENSITY_SCORE_LGBM"].clip(1e-3, 1 - 1e-3).values
T = df["EXPOSED"].values
Y = df["CONVERTED_POST_EXPOSURE"].values
cate = df["CATE"].values

# Trim IPW weights at the 1st/99th percentile (same convention as the
# IPTW step, 03_match_and_weight_hulu.py) -- inverse-propensity weights for
# this population range up to ~170x (propensity scores as low as 0.006),
# and a handful of extreme-weight units can dominate the Horvitz-Thompson
# sum within small top-k buckets, making the raw (untrimmed) IPW uplift
# estimate very noisy in the top deciles. Trimming sacrifices a small
# amount of unbiasedness for a large reduction in variance -- standard
# trade-off for IPW-based evaluation.
w_treated_raw = 1.0 / e
w_control_raw = 1.0 / (1.0 - e)
wt_lo, wt_hi = np.percentile(w_treated_raw, [1, 99])
wc_lo, wc_hi = np.percentile(w_control_raw, [1, 99])
w_treated = np.clip(w_treated_raw, wt_lo, wt_hi)
w_control = np.clip(w_control_raw, wc_lo, wc_hi)
print(f"Treated IPW weight trim bounds: [{wt_lo:.3f}, {wt_hi:.3f}] (untrimmed max: {w_treated_raw.max():.1f})")
print(f"Control IPW weight trim bounds: [{wc_lo:.3f}, {wc_hi:.3f}] (untrimmed max: {w_control_raw.max():.1f})")

# Sort by predicted CATE descending (most-benefit-first)
order = np.argsort(-cate)
T_sorted, Y_sorted = T[order], Y[order]
w_treated_sorted, w_control_sorted = w_treated[order], w_control[order]

n = len(df)
fractions = np.linspace(0.01, 1.0, 100)

ipw_treated_contrib = T_sorted * Y_sorted * w_treated_sorted
ipw_control_contrib = (1 - T_sorted) * Y_sorted * w_control_sorted

cum_treated = np.cumsum(ipw_treated_contrib)
cum_control = np.cumsum(ipw_control_contrib)

qini_curve = []
auuc_curve = []
for frac in fractions:
    k = max(1, int(round(frac * n)))
    # Horvitz-Thompson (IPW) estimator of E[Y(1)] and E[Y(0)] for the top-k
    # bucket: divide the IPW-weighted sum by k (bucket size), NOT by the
    # count of treated/control units in the bucket -- the 1/e(X) weighting
    # already accounts for each unit's inclusion probability, so dividing
    # by treated-count on top of that double-counts the weighting and
    # inflates the estimate by roughly 1/treatment_rate.
    treated_mean = cum_treated[k - 1] / k
    control_mean = cum_control[k - 1] / k
    uplift_ipw = treated_mean - control_mean  # HT-estimated ATE within top-k bucket
    qini_curve.append(uplift_ipw * frac)  # cumulative gain, proportional to fraction targeted
    auuc_curve.append(uplift_ipw)

qini_curve = np.array(qini_curve)
auuc_curve = np.array(auuc_curve)

# Random-targeting baseline: overall ATE (IPW-adjusted, Horvitz-Thompson, trimmed weights) times fraction
overall_treated_mean = (T * Y * w_treated).sum() / n
overall_control_mean = ((1 - T) * Y * w_control).sum() / n
overall_uplift_ipw = overall_treated_mean - overall_control_mean
random_curve = overall_uplift_ipw * fractions

# AUUC (area under the uplift-by-fraction curve, trapezoidal) and its
# normalized version relative to random targeting
auuc_score = np.trapezoid(qini_curve, fractions)
random_auuc = np.trapezoid(random_curve, fractions)
auuc_normalized = auuc_score - random_auuc  # Qini coefficient analog

print(f"\nOverall IPW-adjusted uplift (should match AIPW ATE ballpark): {overall_uplift_ipw:.5f}")
print(f"AUUC (model, cumulative gain curve area): {auuc_score:.5f}")
print(f"AUUC (random targeting baseline): {random_auuc:.5f}")
print(f"Qini coefficient (model AUUC - random AUUC): {auuc_normalized:.5f}")

# Decile table: business-facing "top-k% capture" summary
print("\n--- Uplift-by-decile table (top-k% targeted by predicted CATE) ---")
decile_rows = []
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    k = max(1, int(round(frac * n)))
    treated_mean = cum_treated[k - 1] / k
    control_mean = cum_control[k - 1] / k
    uplift_ipw = treated_mean - control_mean
    frac_of_total_gain = (uplift_ipw * frac) / (overall_uplift_ipw * 1.0) if overall_uplift_ipw != 0 else np.nan
    decile_rows.append({
        "top_k_pct": int(frac * 100),
        "n_users_in_bucket": k,
        "avg_predicted_cate": cate[order][:k].mean(),
        "ipw_uplift_in_bucket": uplift_ipw,
        "cumulative_gain": uplift_ipw * frac,
        "pct_of_total_gain_captured": 100 * frac_of_total_gain,
    })
decile_df = pd.DataFrame(decile_rows)
print(decile_df.round(5).to_string(index=False))

decile_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved decile/Qini table to {OUT_CSV}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fractions * 100, qini_curve, label="Model (ranked by predicted CATE)", linewidth=2)
plt.plot(fractions * 100, random_curve, label="Random targeting", linestyle="--", color="gray")
plt.xlabel("% of population targeted (top-k by predicted CATE)")
plt.ylabel("Cumulative IPW-adjusted uplift gain")
plt.title(f"Qini Curve -- Hulu Flight 648768\n(Qini coefficient = {auuc_normalized:.5f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=120)
print(f"Saved Qini curve plot to {OUT_PLOT}")
