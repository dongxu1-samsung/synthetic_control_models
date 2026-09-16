"""
Step 5: Lift estimation (Hulu campaign 291260, flight 648768).

Computes incremental conversion lift under three approaches:
  (a) Naive/raw (unadjusted) -- for reference only, expected to be biased
  (b) 1:1 caliper-matched -- paired difference-in-means + paired t-test
  (c) IPTW (stabilized, trimmed) -- weighted difference-in-means + bootstrap CI
"""
import numpy as np
import pandas as pd
from scipy import stats

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored_hulu.parquet"
MATCHED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/matched_pairs_hulu.parquet"
WEIGHTED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/iptw_weighted_hulu.parquet"

N_BOOTSTRAP = 2000
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

results = {}

# ------------------------------------------------------------------
# (a) Naive / unadjusted
# ------------------------------------------------------------------
raw = pd.read_parquet(RAW_PATH)
rate_t_raw = raw.loc[raw.EXPOSED == 1, "CONVERTED_POST_EXPOSURE"].mean()
rate_c_raw = raw.loc[raw.EXPOSED == 0, "CONVERTED_POST_EXPOSURE"].mean()
lift_raw = rate_t_raw - rate_c_raw
n_t_raw = (raw.EXPOSED == 1).sum()
n_c_raw = (raw.EXPOSED == 0).sum()

se_raw = np.sqrt(rate_t_raw * (1 - rate_t_raw) / n_t_raw + rate_c_raw * (1 - rate_c_raw) / n_c_raw)
z_raw = lift_raw / se_raw
p_raw = 2 * (1 - stats.norm.cdf(abs(z_raw)))

results["naive_unadjusted"] = {
    "n_treated": int(n_t_raw), "n_control": int(n_c_raw),
    "conv_rate_treated": rate_t_raw, "conv_rate_control": rate_c_raw,
    "abs_lift": lift_raw, "rel_lift_pct": 100 * lift_raw / rate_c_raw,
    "se": se_raw, "z": z_raw, "p_value": p_raw,
}

# ------------------------------------------------------------------
# (b) 1:1 caliper-matched -- paired analysis
# ------------------------------------------------------------------
matched = pd.read_parquet(MATCHED_PATH)
m_treated = matched[matched.GROUP == "treated"].sort_values("PAIR_ID").reset_index(drop=True)
m_control = matched[matched.GROUP == "control"].sort_values("PAIR_ID").reset_index(drop=True)

conv_t = m_treated["CONVERTED_POST_EXPOSURE"].values
conv_c = m_control["CONVERTED_POST_EXPOSURE"].values
paired_diff = conv_t - conv_c

lift_matched = paired_diff.mean()
rate_t_matched = conv_t.mean()
rate_c_matched = conv_c.mean()

t_stat, p_matched = stats.ttest_rel(conv_t, conv_c)
se_matched = paired_diff.std(ddof=1) / np.sqrt(len(paired_diff))
ci_lo_matched = lift_matched - 1.96 * se_matched
ci_hi_matched = lift_matched + 1.96 * se_matched

results["matched_1to1_caliper"] = {
    "n_pairs": len(paired_diff),
    "conv_rate_treated": rate_t_matched, "conv_rate_control": rate_c_matched,
    "abs_lift": lift_matched, "rel_lift_pct": 100 * lift_matched / rate_c_matched,
    "se": se_matched, "ci_95_lo": ci_lo_matched, "ci_95_hi": ci_hi_matched,
    "t_stat": t_stat, "p_value": p_matched,
}

# ------------------------------------------------------------------
# (c) IPTW (stabilized, trimmed) -- weighted diff-in-means + bootstrap CI
# ------------------------------------------------------------------
weighted = pd.read_parquet(WEIGHTED_PATH)
w_treated = weighted[weighted.EXPOSED == 1]
w_control = weighted[weighted.EXPOSED == 0]

def weighted_rate(vals, w):
    return np.average(vals, weights=w)

rate_t_iptw = weighted_rate(w_treated["CONVERTED_POST_EXPOSURE"].values, w_treated["IPTW_WEIGHT_TRIMMED"].values)
rate_c_iptw = weighted_rate(w_control["CONVERTED_POST_EXPOSURE"].values, w_control["IPTW_WEIGHT_TRIMMED"].values)
lift_iptw = rate_t_iptw - rate_c_iptw

t_vals = w_treated["CONVERTED_POST_EXPOSURE"].values
t_w = w_treated["IPTW_WEIGHT_TRIMMED"].values
c_vals = w_control["CONVERTED_POST_EXPOSURE"].values
c_w = w_control["IPTW_WEIGHT_TRIMMED"].values

n_t, n_c = len(t_vals), len(c_vals)
boot_lifts = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    idx_t = rng.integers(0, n_t, n_t)
    idx_c = rng.integers(0, n_c, n_c)
    r_t = np.average(t_vals[idx_t], weights=t_w[idx_t])
    r_c = np.average(c_vals[idx_c], weights=c_w[idx_c])
    boot_lifts[b] = r_t - r_c

ci_lo_iptw, ci_hi_iptw = np.percentile(boot_lifts, [2.5, 97.5])
p_iptw_boot = 2 * min((boot_lifts <= 0).mean(), (boot_lifts >= 0).mean())

results["iptw_stabilized_trimmed"] = {
    "n_treated": int(n_t), "n_control": int(n_c),
    "conv_rate_treated": rate_t_iptw, "conv_rate_control": rate_c_iptw,
    "abs_lift": lift_iptw, "rel_lift_pct": 100 * lift_iptw / rate_c_iptw,
    "ci_95_lo": ci_lo_iptw, "ci_95_hi": ci_hi_iptw,
    "bootstrap_p_value_two_sided": p_iptw_boot,
    "n_bootstrap": N_BOOTSTRAP,
}

# ------------------------------------------------------------------
# Print summary table
# ------------------------------------------------------------------
print("=" * 90)
print("LIFT ESTIMATION SUMMARY -- Hulu Flight 648768 (Campaign 291260)")
print("=" * 90)

for method, r in results.items():
    print(f"\n--- {method} ---")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

print("\n" + "=" * 90)
print("HEADLINE COMPARISON")
print("=" * 90)
summary_df = pd.DataFrame({
    "method": ["naive_unadjusted", "matched_1to1_caliper", "iptw_stabilized_trimmed"],
    "conv_rate_treated": [
        results["naive_unadjusted"]["conv_rate_treated"],
        results["matched_1to1_caliper"]["conv_rate_treated"],
        results["iptw_stabilized_trimmed"]["conv_rate_treated"],
    ],
    "conv_rate_control": [
        results["naive_unadjusted"]["conv_rate_control"],
        results["matched_1to1_caliper"]["conv_rate_control"],
        results["iptw_stabilized_trimmed"]["conv_rate_control"],
    ],
    "abs_lift_pp": [
        100 * results["naive_unadjusted"]["abs_lift"],
        100 * results["matched_1to1_caliper"]["abs_lift"],
        100 * results["iptw_stabilized_trimmed"]["abs_lift"],
    ],
    "rel_lift_pct": [
        results["naive_unadjusted"]["rel_lift_pct"],
        results["matched_1to1_caliper"]["rel_lift_pct"],
        results["iptw_stabilized_trimmed"]["rel_lift_pct"],
    ],
})
print(summary_df.round(4).to_string(index=False))

summary_df.to_csv(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_hulu.csv", index=False
)
print("\nSaved lift summary to /Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_summary_hulu.csv")
