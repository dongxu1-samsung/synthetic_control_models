"""
Step 6: Robustness checks (Hulu campaign 291260, flight 648768).

(a) Placebo test: split the CONTROL pool into two pseudo-groups (both truly
    unexposed) and run the same matching pipeline. If the pipeline is sound,
    measured "lift" between two unexposed groups should be ~0.

(b) Investigate matched vs. IPTW divergence: compare effective sample size
    for matching (given control reuse) and check whether restricting
    to unique matched controls only changes the estimate materially.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored_hulu.parquet"
MATCHED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/matched_pairs_hulu.parquet"

rng = np.random.default_rng(42)

# ==================================================================
# (a) Placebo test
# ==================================================================
raw = pd.read_parquet(RAW_PATH)
control_pool = raw[raw.EXPOSED == 0].copy().reset_index(drop=True)
print(f"Control pool size: {len(control_pool)}")

shuffled_idx = rng.permutation(len(control_pool))
half = len(control_pool) // 2
pseudo_treated = control_pool.iloc[shuffled_idx[:half]].reset_index(drop=True)
pseudo_control = control_pool.iloc[shuffled_idx[half:]].reset_index(drop=True)

print(f"Pseudo-treated: {len(pseudo_treated)}, Pseudo-control: {len(pseudo_control)}")

rate_pt = pseudo_treated["CONVERTED_POST_EXPOSURE"].mean()
rate_pc = pseudo_control["CONVERTED_POST_EXPOSURE"].mean()
placebo_naive_lift = rate_pt - rate_pc
se_placebo = np.sqrt(
    rate_pt * (1 - rate_pt) / len(pseudo_treated) + rate_pc * (1 - rate_pc) / len(pseudo_control)
)
z_placebo = placebo_naive_lift / se_placebo
p_placebo = 2 * (1 - stats.norm.cdf(abs(z_placebo)))

print(f"\n--- Placebo test (naive, unmatched) ---")
print(f"Pseudo-treated conv rate: {rate_pt:.5f}")
print(f"Pseudo-control conv rate: {rate_pc:.5f}")
print(f"Placebo lift (should be ~0): {placebo_naive_lift:.5f} ({100*placebo_naive_lift/rate_pc:.2f}% relative)")
print(f"z={z_placebo:.3f}, p={p_placebo:.4f}")

# Placebo test with matching (re-fit NN matching on propensity score within
# the two pseudo-groups)
nn = NearestNeighbors(n_neighbors=1)
nn.fit(pseudo_control[["PROPENSITY_SCORE"]].values)
distances, indices = nn.kneighbors(pseudo_treated[["PROPENSITY_SCORE"]].values)
distances = distances.ravel()
indices = indices.ravel()
within_caliper = distances <= 0.01

matched_pt = pseudo_treated[within_caliper].reset_index(drop=True)
matched_pc = pseudo_control.iloc[indices[within_caliper]].reset_index(drop=True)

conv_pt = matched_pt["CONVERTED_POST_EXPOSURE"].values
conv_pc = matched_pc["CONVERTED_POST_EXPOSURE"].values
paired_diff_placebo = conv_pt - conv_pc
placebo_matched_lift = paired_diff_placebo.mean()
t_stat_placebo, p_matched_placebo = stats.ttest_rel(conv_pt, conv_pc)

print(f"\n--- Placebo test (1:1 caliper matched) ---")
print(f"Matched pairs: {len(paired_diff_placebo)} / {len(pseudo_treated)} "
      f"({100*within_caliper.sum()/len(pseudo_treated):.1f}% matched)")
print(f"Placebo matched lift (should be ~0): {placebo_matched_lift:.5f}")
print(f"t={t_stat_placebo:.3f}, p={p_matched_placebo:.4f}")

# ==================================================================
# (b) Matched vs IPTW divergence investigation
# ==================================================================
matched = pd.read_parquet(MATCHED_PATH)
m_control = matched[matched.GROUP == "control"]
m_treated = matched[matched.GROUP == "treated"]

reuse_counts = m_control["PSID"].value_counts()
print(f"\n--- Matching reuse diagnostics ---")
print(f"Unique control PSIDs used: {m_control['PSID'].nunique()}")
print(f"Total matched pairs: {len(m_control)}")
print(f"Avg reuse per unique control: {len(m_control) / m_control['PSID'].nunique():.2f}")
print(f"Max reuse (single control matched to N treated): {reuse_counts.max()}")
print(f"Reuse distribution (percentiles): {np.percentile(reuse_counts.values, [50, 90, 99, 100])}")

control_conv_by_unique = m_control.drop_duplicates(subset="PSID")
print(f"\nConversion rate among UNIQUE matched controls (no reuse weighting): "
      f"{control_conv_by_unique['CONVERTED_POST_EXPOSURE'].mean():.5f}")
print(f"Conversion rate among matched controls AS USED (with reuse, i.e. what "
      f"the paired lift estimate actually reflects): {m_control['CONVERTED_POST_EXPOSURE'].mean():.5f}")

full_control = pd.read_parquet(RAW_PATH)
full_control = full_control[full_control.EXPOSED == 0]
print(f"\nFull control pool propensity score: mean={full_control['PROPENSITY_SCORE'].mean():.4f}, "
      f"median={full_control['PROPENSITY_SCORE'].median():.4f}")
print(f"Unique matched control propensity score: mean={control_conv_by_unique['PROPENSITY_SCORE'].mean():.4f}, "
      f"median={control_conv_by_unique['PROPENSITY_SCORE'].median():.4f}")
print(f"Treated pool propensity score: mean={m_treated['PROPENSITY_SCORE'].mean():.4f}, "
      f"median={m_treated['PROPENSITY_SCORE'].median():.4f}")
