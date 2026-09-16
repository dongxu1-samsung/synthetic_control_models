"""
Step 7: Final lift summary, consolidating all validated results.

Hulu campaign 291260, flight 648768.
"""
import pandas as pd

final = pd.DataFrame({
    "method": [
        "naive_unadjusted",
        "matched_1to1_caliper (derandomized, avg of 10 repeats)",
        "iptw_stabilized_trimmed",
    ],
    "conv_rate_treated": [0.06674, 0.06674, 0.06574],
    "conv_rate_control": [0.03217, 0.03484, 0.03249],
    "abs_lift_pp": [3.4565, 3.190, 3.325],
    "rel_lift_pct": [107.4355, 91.5453, 102.3243],
    "placebo_test_passed": ["N/A (reference only)", "YES (p=0.84 single-run; p=0.47 derandomized)", "YES (p=0.76, refit-on-pseudo-split)"],
    "notes": [
        "Biased -- RTB auction win is not random, naive comparison overstates lift",
        "Single-run estimate (91.55% rel. lift) confirmed by 10x jittered re-run (avg abs lift 0.03192pp, range 0.03153-0.03221); tie-breaking risk checked as a precaution (root cause of a bug found on the Apple TV pipeline) -- not actually present here, matching was already unbiased in a single run",
        "Clean covariate balance (max SMD 0.052 < 0.1 threshold), placebo-validated with propensity model refit on the pseudo-split",
    ],
})

print(final.to_string(index=False))
final.to_csv("/Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_final_hulu.csv", index=False)
print("\nSaved to /Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_final_hulu.csv")
