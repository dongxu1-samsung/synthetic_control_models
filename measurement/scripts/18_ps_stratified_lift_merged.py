"""
Step 18: Propensity-decile stratified lift breakdown on the MERGED dataset
-- same diagnostic as 09_ps_stratified_lift.py, needed because matched
(-7.6%) and IPTW (+9.4%) sign-diverge here (worse than the original
non-merged pipeline's 9.2% vs 21.6%, likely because the richer S3 features
give the propensity model much better discrimination -- AUC 0.79 vs 0.72 --
so propensity now captures more of the heterogeneity in conversion
likelihood, and reused high-propensity controls in the matched design pull
the matched-control rate up further).
"""
import numpy as np
import pandas as pd

IN_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_iptw_weighted.parquet"

df = pd.read_parquet(IN_PATH)
print("Loaded:", df.shape)

df["ps_decile"] = pd.qcut(df["PROPENSITY_SCORE"], 10, labels=False, duplicates="drop")

rows = []
for d in sorted(df["ps_decile"].unique()):
    sub = df[df["ps_decile"] == d]
    t = sub[sub.EXPOSED == 1]
    c = sub[sub.EXPOSED == 0]
    if len(t) == 0 or len(c) == 0:
        continue
    rate_t = t["CONVERTED_POST_EXPOSURE"].mean()
    rate_c = c["CONVERTED_POST_EXPOSURE"].mean()
    abs_lift = rate_t - rate_c
    rel_lift = 100 * abs_lift / rate_c if rate_c > 0 else np.nan
    ps_range = (sub["PROPENSITY_SCORE"].min(), sub["PROPENSITY_SCORE"].max())
    rows.append({
        "decile": d, "ps_min": ps_range[0], "ps_max": ps_range[1],
        "n_treated": len(t), "n_control": len(c),
        "conv_rate_treated": rate_t, "conv_rate_control": rate_c,
        "abs_lift": abs_lift, "rel_lift_pct": rel_lift,
    })

decile_df = pd.DataFrame(rows)
pd.set_option("display.width", 150)
print("\nLift by propensity-score decile:")
print(decile_df.round(4).to_string(index=False))

# Reconstruct the two competing weighting schemes
treated_weighted_lift = np.average(decile_df["abs_lift"], weights=decile_df["n_treated"])
equal_weighted_lift = decile_df["abs_lift"].mean()

overall_control_rate = np.average(decile_df["conv_rate_control"], weights=decile_df["n_treated"])
print(f"\nTreated-population-weighted avg abs lift: {treated_weighted_lift:.5f} "
      f"({100*treated_weighted_lift/overall_control_rate:.2f}% rel, mirrors MATCHING-style estimate)")
print(f"Equal-weighted avg abs lift across deciles: {equal_weighted_lift:.5f} "
      f"({100*equal_weighted_lift/overall_control_rate:.2f}% rel, mirrors IPTW-style estimate)")

decile_df.to_csv(
    "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_lift_by_ps_decile.csv", index=False
)
print("\nSaved to merged_lift_by_ps_decile.csv")
