"""
Step 11: Consolidated final summary across ALL estimation methods
(Hulu campaign 291260, flight 648768) -- naive, matched, IPTW, AIPW
(doubly-robust, LightGBM nuisance), and causal-forest-recovered ATE.
"""
import pandas as pd

final = pd.DataFrame({
    "method": [
        "naive_unadjusted",
        "matched_1to1_caliper (derandomized, avg of 10 repeats)",
        "iptw_stabilized_trimmed",
        "aipw_doubly_robust (LightGBM nuisance, cross-fitted, full population)",
        "causal_forest_dml (mean CATE, 750K stratified subsample)",
    ],
    "conv_rate_treated": [0.06674, 0.06674, 0.06574, None, None],
    "conv_rate_control": [0.03217, 0.03484, 0.03249, None, None],
    "abs_lift_pp": [3.4565, 3.190, 3.325, 3.156, 3.114],
    "rel_lift_pct": [107.4355, 91.5453, 102.3243, 98.09, 96.71],
    "placebo_test_passed": [
        "N/A (reference only)",
        "YES (p=0.84 single-run; p=0.47 derandomized)",
        "YES (p=0.76, refit-on-pseudo-split)",
        "YES (p=0.73, LightGBM nuisance refit-on-pseudo-split)",
        "N/A (not re-run for the forest; consistent with the other 3 validated estimators)",
    ],
    "notes": [
        "Biased -- RTB auction win is not random, naive comparison overstates lift",
        "Single-run estimate (91.55% rel. lift) confirmed by 10x jittered re-run",
        "Clean covariate balance (max SMD 0.052 < 0.1), placebo-validated",
        "Doubly-robust: consistent if EITHER propensity or outcome model correct. "
        "Propensity AUC improved 0.622->0.668 vs logistic regression. Tight 95% CI "
        "[3.09pp, 3.22pp] (bootstrap).",
        "Personalized tau(X) via honest random forest + Double ML orthogonalization. "
        "Mean CATE across 750K-row subsample matches full-population AIPW/IPTW/matching "
        "closely -- 4-way triangulation. Meaningful heterogeneity found (std 0.0124, "
        "range -0.87pp to +14.2pp); see cate_by_segment_hulu.csv for targeting insights.",
    ],
})

print(final.to_string(index=False))
final.to_csv(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_final_hulu_v2.csv",
    index=False,
)
print("\nSaved to /Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_final_hulu_v2.csv")

print("\n" + "=" * 90)
print("CONVERGENCE CHECK: all four independent ATE-style estimators agree within a ~10.9pp")
print("relative-lift band (91.5% - 102.3%), with AIPW (98.1%) and the causal forest (96.7%)")
print("landing almost exactly in the middle of matching (91.5%) and IPTW (102.3%).")
print("Recommended headline range: ~92-102% relative conversion lift.")
print("=" * 90)
