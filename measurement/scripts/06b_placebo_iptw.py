import numpy as np
import pandas as pd
from scipy import stats

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"
rng = np.random.default_rng(42)

raw = pd.read_parquet(RAW_PATH)
control_pool = raw[raw.EXPOSED == 0].copy().reset_index(drop=True)

shuffled_idx = rng.permutation(len(control_pool))
half = len(control_pool) // 2
pseudo_treated = control_pool.iloc[shuffled_idx[:half]].reset_index(drop=True)
pseudo_control = control_pool.iloc[shuffled_idx[half:]].reset_index(drop=True)

# IPTW placebo: refit stabilized weights using the SAME propensity score
# (already computed) but treat/control defined by the pseudo split
p = pseudo_treated["PROPENSITY_SCORE"].clip(1e-4, 1 - 1e-4)
pc = pseudo_control["PROPENSITY_SCORE"].clip(1e-4, 1 - 1e-4)

# marginal "treat" probability under pseudo split is 0.5 by construction
sw_t = 0.5 / p
sw_c = 0.5 / (1 - pc)

lo_t, hi_t = np.percentile(sw_t, [1, 99])
lo_c, hi_c = np.percentile(sw_c, [1, 99])
sw_t_trim = np.clip(sw_t, lo_t, hi_t)
sw_c_trim = np.clip(sw_c, lo_c, hi_c)

rate_pt = np.average(pseudo_treated["CONVERTED_POST_EXPOSURE"].values, weights=sw_t_trim)
rate_pc = np.average(pseudo_control["CONVERTED_POST_EXPOSURE"].values, weights=sw_c_trim)
placebo_iptw_lift = rate_pt - rate_pc

# bootstrap CI
t_vals = pseudo_treated["CONVERTED_POST_EXPOSURE"].values
c_vals = pseudo_control["CONVERTED_POST_EXPOSURE"].values
n_t, n_c = len(t_vals), len(c_vals)
N_BOOT = 1000
boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx_t = rng.integers(0, n_t, n_t)
    idx_c = rng.integers(0, n_c, n_c)
    r_t = np.average(t_vals[idx_t], weights=sw_t_trim[idx_t])
    r_c = np.average(c_vals[idx_c], weights=sw_c_trim[idx_c])
    boot[b] = r_t - r_c
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
p_boot = 2 * min((boot <= 0).mean(), (boot >= 0).mean())

print("--- Placebo test (IPTW, stabilized+trimmed) ---")
print(f"Pseudo-treated weighted conv rate: {rate_pt:.5f}")
print(f"Pseudo-control weighted conv rate: {rate_pc:.5f}")
print(f"Placebo IPTW lift (should be ~0): {placebo_iptw_lift:.5f}")
print(f"95% CI: [{ci_lo:.5f}, {ci_hi:.5f}], bootstrap p={p_boot:.4f}")
