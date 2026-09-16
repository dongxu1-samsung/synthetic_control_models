"""
Diagnostic #1b: Propensity-stratified lift breakdown.

Directly tests the effect-heterogeneity hypothesis from the population-scope
check: does lift vary systematically across propensity-score strata? If so,
that explains why matching (concentrated near the treated distribution,
higher propensity) and IPTW (representative across the full range) diverge.
"""
import numpy as np
import pandas as pd

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"

df = pd.read_parquet(RAW_PATH)

# Common support trim (same as main pipeline)
p_exposed = df.loc[df.EXPOSED == 1, "PROPENSITY_SCORE"]
p_control = df.loc[df.EXPOSED == 0, "PROPENSITY_SCORE"]
lower = max(p_exposed.min(), p_control.min())
upper = min(p_exposed.max(), p_control.max())
df = df[(df.PROPENSITY_SCORE >= lower) & (df.PROPENSITY_SCORE <= upper)].copy()

# Decile bins on the pooled propensity score
df["PS_DECILE"] = pd.qcut(df["PROPENSITY_SCORE"], 10, labels=False, duplicates="drop")

rows = []
for decile in sorted(df["PS_DECILE"].unique()):
    sub = df[df["PS_DECILE"] == decile]
    t = sub[sub.EXPOSED == 1]
    c = sub[sub.EXPOSED == 0]
    if len(t) == 0 or len(c) == 0:
        continue
    rate_t = t["CONVERTED_POST_EXPOSURE"].mean()
    rate_c = c["CONVERTED_POST_EXPOSURE"].mean()
    lift = rate_t - rate_c
    rel_lift = 100 * lift / rate_c if rate_c > 0 else np.nan
    rows.append({
        "ps_decile": decile,
        "ps_range": f"[{sub['PROPENSITY_SCORE'].min():.3f}, {sub['PROPENSITY_SCORE'].max():.3f}]",
        "n_treated": len(t),
        "n_control": len(c),
        "conv_rate_treated": rate_t,
        "conv_rate_control": rate_c,
        "abs_lift": lift,
        "rel_lift_pct": rel_lift,
    })

result = pd.DataFrame(rows)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 150)
print("Lift by propensity-score decile:\n")
print(result.round(4).to_string(index=False))

# Weighted overall (pooled across deciles, treated-count-weighted -- this
# approximates what matching effectively does, since matched treated units
# are distributed like the full treated population)
overall_treated_weighted_lift = np.average(result["abs_lift"], weights=result["n_treated"])
overall_treated_weighted_control_rate = np.average(result["conv_rate_control"], weights=result["n_treated"])
print(f"\nTreated-distribution-weighted avg lift (mimics matching's implicit weighting): "
      f"{overall_treated_weighted_lift:.5f} "
      f"({100*overall_treated_weighted_lift/overall_treated_weighted_control_rate:.2f}% relative)")

# Equal-weighted across deciles (mimics IPTW's more equal-representation goal)
equal_weighted_lift = result["abs_lift"].mean()
equal_weighted_control_rate = result["conv_rate_control"].mean()
print(f"Equal-weighted avg lift across deciles (mimics IPTW's flatter weighting): "
      f"{equal_weighted_lift:.5f} "
      f"({100*equal_weighted_lift/equal_weighted_control_rate:.2f}% relative)")

result.to_csv("/Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_by_ps_decile.csv", index=False)
print("\nSaved to /Users/dongd1.xu/Documents/hermes_agent/measurement/data/lift_by_ps_decile.csv")
