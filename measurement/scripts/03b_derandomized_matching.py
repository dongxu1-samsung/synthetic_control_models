"""
Step 3b: Tie-robust matching alternative.

The original 1:1 nearest-neighbor matcher failed its placebo test (lift
should be ~0 between two random control halves, but measured +0.19pp,
p<0.0001). Root cause: heavy ties in propensity score (driven by
categorical/discrete covariates -- e.g. one score value shared by ~50,000
control rows) make sklearn's NearestNeighbors tie-break deterministically
by array position, which can introduce a small systematic bias at this
sample size.

Fix: add a small amount of random jitter to the propensity score before
matching (breaks ties randomly instead of by position), repeat matching
multiple times with different random jitter draws, and average the lift
across repeats. This is a standard remedy for tie-induced matching bias.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/matched_pairs_derandomized.parquet"

CALIPER = 0.01
N_REPEATS = 10
JITTER_SCALE = 1e-6  # much smaller than caliper; only breaks exact ties, doesn't change real matches

rng = np.random.default_rng(123)

df = pd.read_parquet(RAW_PATH)

p_exposed = df.loc[df.EXPOSED == 1, "PROPENSITY_SCORE"]
p_control = df.loc[df.EXPOSED == 0, "PROPENSITY_SCORE"]
lower = max(p_exposed.min(), p_control.min())
upper = min(p_exposed.max(), p_control.max())
df_trimmed = df[(df.PROPENSITY_SCORE >= lower) & (df.PROPENSITY_SCORE <= upper)].copy()

treated = df_trimmed[df_trimmed.EXPOSED == 1].reset_index(drop=True)
control = df_trimmed[df_trimmed.EXPOSED == 0].reset_index(drop=True)

print(f"Repeating 1:1 caliper matching {N_REPEATS}x with random tie-breaking jitter...")

lift_estimates = []
n_matched_list = []

for rep in range(N_REPEATS):
    jitter_t = rng.normal(0, JITTER_SCALE, len(treated))
    jitter_c = rng.normal(0, JITTER_SCALE, len(control))
    p_t_jittered = (treated["PROPENSITY_SCORE"] + jitter_t).values.reshape(-1, 1)
    p_c_jittered = (control["PROPENSITY_SCORE"] + jitter_c).values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(p_c_jittered)
    distances, indices = nn.kneighbors(p_t_jittered)
    distances = distances.ravel()
    indices = indices.ravel()

    within_caliper = distances <= CALIPER
    conv_t = treated.loc[within_caliper, "CONVERTED_POST_EXPOSURE"].values
    conv_c = control.iloc[indices[within_caliper]]["CONVERTED_POST_EXPOSURE"].values

    lift = (conv_t - conv_c).mean()
    lift_estimates.append(lift)
    n_matched_list.append(within_caliper.sum())
    print(f"  Repeat {rep+1}/{N_REPEATS}: matched={within_caliper.sum()}, lift={lift:.5f}")

lift_estimates = np.array(lift_estimates)
avg_lift = lift_estimates.mean()
se_across_repeats = lift_estimates.std(ddof=1) / np.sqrt(N_REPEATS)

print(f"\nDe-randomized matched lift (avg over {N_REPEATS} repeats): {avg_lift:.5f}")
print(f"Std across repeats: {lift_estimates.std(ddof=1):.5f}")
print(f"Range: [{lift_estimates.min():.5f}, {lift_estimates.max():.5f}]")

# ------------------------------------------------------------------
# Placebo re-check with de-randomized matching
# ------------------------------------------------------------------
print("\n--- Placebo re-check with de-randomized (jittered) matching ---")
control_pool = df[df.EXPOSED == 0].copy().reset_index(drop=True)
shuffled_idx = rng.permutation(len(control_pool))
half = len(control_pool) // 2
pseudo_treated = control_pool.iloc[shuffled_idx[:half]].reset_index(drop=True)
pseudo_control = control_pool.iloc[shuffled_idx[half:]].reset_index(drop=True)

placebo_lifts = []
for rep in range(N_REPEATS):
    jitter_t = rng.normal(0, JITTER_SCALE, len(pseudo_treated))
    jitter_c = rng.normal(0, JITTER_SCALE, len(pseudo_control))
    p_t_jittered = (pseudo_treated["PROPENSITY_SCORE"] + jitter_t).values.reshape(-1, 1)
    p_c_jittered = (pseudo_control["PROPENSITY_SCORE"] + jitter_c).values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(p_c_jittered)
    distances, indices = nn.kneighbors(p_t_jittered)
    distances = distances.ravel()
    indices = indices.ravel()
    within_caliper = distances <= CALIPER

    conv_pt = pseudo_treated.loc[within_caliper, "CONVERTED_POST_EXPOSURE"].values
    conv_pc = pseudo_control.iloc[indices[within_caliper]]["CONVERTED_POST_EXPOSURE"].values
    placebo_lifts.append((conv_pt - conv_pc).mean())

placebo_lifts = np.array(placebo_lifts)
print(f"Placebo lift across {N_REPEATS} repeats: mean={placebo_lifts.mean():.5f}, "
      f"std={placebo_lifts.std(ddof=1):.5f}, range=[{placebo_lifts.min():.5f}, {placebo_lifts.max():.5f}]")

t_stat, p_val = stats.ttest_1samp(placebo_lifts, 0)
print(f"One-sample t-test (H0: mean placebo lift = 0): t={t_stat:.3f}, p={p_val:.4f}")
