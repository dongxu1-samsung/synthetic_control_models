"""
Step 3: Propensity-based bias correction.

Hulu campaign 291260, flight 648768.

Two complementary approaches, both computed and saved:
  (a) 1:1 nearest-neighbor caliper matching on propensity score
      (interpretable, standard for lift reporting)
  (b) Inverse Probability of Treatment Weighting (IPTW), stabilized
      (uses full sample, more statistically efficient)

Common-support trimming is applied first (drop units outside the overlapping
propensity range) to avoid extrapolation.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored_hulu.parquet"
OUT_MATCHED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/matched_pairs_hulu.parquet"
OUT_WEIGHTED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/iptw_weighted_hulu.parquet"

df = pd.read_parquet(IN_PATH)
print("Loaded:", df.shape)

# ------------------------------------------------------------------
# Common support trimming: restrict to the overlapping propensity range
# between exposed and control groups (standard practice before matching)
# ------------------------------------------------------------------
p_exposed = df.loc[df.EXPOSED == 1, "PROPENSITY_SCORE"]
p_control = df.loc[df.EXPOSED == 0, "PROPENSITY_SCORE"]

lower = max(p_exposed.min(), p_control.min())
upper = min(p_exposed.max(), p_control.max())
print(f"Common support range: [{lower:.4f}, {upper:.4f}]")

before_n = len(df)
df_trimmed = df[(df.PROPENSITY_SCORE >= lower) & (df.PROPENSITY_SCORE <= upper)].copy()
print(f"Rows before trimming: {before_n}, after: {len(df_trimmed)} "
      f"({100*(before_n - len(df_trimmed))/before_n:.2f}% dropped)")

# ==================================================================
# (a) 1:1 nearest-neighbor caliper matching on propensity score
# ==================================================================
CALIPER = 0.01  # same caliper as the Apple TV pipeline, for consistency

treated = df_trimmed[df_trimmed.EXPOSED == 1].reset_index(drop=True)
control = df_trimmed[df_trimmed.EXPOSED == 0].reset_index(drop=True)

print(f"\nMatching {len(treated)} treated to {len(control)} control candidates "
      f"(caliper={CALIPER})...")

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["PROPENSITY_SCORE"]].values)
distances, indices = nn.kneighbors(treated[["PROPENSITY_SCORE"]].values)

distances = distances.ravel()
indices = indices.ravel()

within_caliper = distances <= CALIPER
print(f"Treated units matched within caliper: {within_caliper.sum()} / {len(treated)} "
      f"({100*within_caliper.sum()/len(treated):.1f}%)")

matched_treated = treated[within_caliper].copy()
matched_treated["MATCH_DISTANCE"] = distances[within_caliper]
matched_control_idx = indices[within_caliper]
matched_control = control.iloc[matched_control_idx].copy().reset_index(drop=True)
matched_treated = matched_treated.reset_index(drop=True)

matched_treated["PAIR_ID"] = np.arange(len(matched_treated))
matched_control["PAIR_ID"] = np.arange(len(matched_control))

# Matching WITH replacement (same convention as the Apple TV pipeline) --
# a given control row can serve as the nearest neighbor for more than one
# treated row.
n_unique_controls_used = matched_control["PSID"].nunique()
print(f"Unique control units used: {n_unique_controls_used} / {len(matched_control)} matches "
      f"({'with' if n_unique_controls_used < len(matched_control) else 'without'} reuse)")

matched_pairs = pd.concat(
    [
        matched_treated.assign(GROUP="treated"),
        matched_control.assign(GROUP="control"),
    ],
    ignore_index=True,
)
matched_pairs.to_parquet(OUT_MATCHED_PATH, index=False)
print(f"Saved matched pairs to {OUT_MATCHED_PATH} ({len(matched_pairs)} rows)")

# ==================================================================
# (b) Stabilized IPTW (uses full trimmed sample)
# ==================================================================
p = df_trimmed["PROPENSITY_SCORE"].clip(1e-4, 1 - 1e-4)  # avoid extreme weights
treat = df_trimmed["EXPOSED"]

# Marginal treatment probability, for stabilization
p_treat_marginal = treat.mean()

# Stabilized weights: SW = P(T=t) / P(T=t | X) -- reduces variance vs. plain IPTW
sw = np.where(
    treat == 1,
    p_treat_marginal / p,
    (1 - p_treat_marginal) / (1 - p),
)
df_trimmed["IPTW_WEIGHT"] = sw

# Weight trimming at 1st/99th percentile to limit influence of extreme weights
lo, hi = np.percentile(sw, [1, 99])
df_trimmed["IPTW_WEIGHT_TRIMMED"] = np.clip(sw, lo, hi)

print("\nIPTW weight distribution (stabilized):")
print(df_trimmed.groupby("EXPOSED")["IPTW_WEIGHT"].describe())
print(f"\n1st/99th percentile trim bounds: [{lo:.3f}, {hi:.3f}]")

df_trimmed.to_parquet(OUT_WEIGHTED_PATH, index=False)
print(f"Saved IPTW-weighted data to {OUT_WEIGHTED_PATH} ({len(df_trimmed)} rows)")
