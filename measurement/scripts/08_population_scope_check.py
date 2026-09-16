"""
Diagnostic #1: Check if matched-vs-IPTW divergence is due to population
scope rather than bias.

Matching only keeps treated units with a close control match (implicitly
restricts to a subpopulation with good overlap). IPTW uses the full
trimmed sample. This script recomputes IPTW restricted to ONLY the matched
treated units (and their matched controls' propensity-weighted estimate)
to see if the two methods converge once they're estimating the same
population.
"""
import numpy as np
import pandas as pd

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_scored.parquet"
MATCHED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/matched_pairs.parquet"
WEIGHTED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/iptw_weighted.parquet"

matched = pd.read_parquet(MATCHED_PATH)
weighted = pd.read_parquet(WEIGHTED_PATH)

m_treated = matched[matched.GROUP == "treated"]
m_control = matched[matched.GROUP == "control"]
matched_treated_psids = set(m_treated["PSID"])
matched_control_psids = set(m_control["PSID"])

print(f"Matched treated units: {len(matched_treated_psids)}")
print(f"Unique matched control units: {len(matched_control_psids)}")

# ------------------------------------------------------------------
# (a) IPTW restricted to ONLY the matched treated units + ONLY the
#     controls that were used at least once as a match
# ------------------------------------------------------------------
w_treated_full = weighted[weighted.EXPOSED == 1]
w_control_full = weighted[weighted.EXPOSED == 0]

w_treated_restricted = w_treated_full[w_treated_full["PSID"].isin(matched_treated_psids)]
w_control_restricted = w_control_full[w_control_full["PSID"].isin(matched_control_psids)]

print(f"\nIPTW treated pool restricted to matched treated: {len(w_treated_restricted)} "
      f"(full IPTW treated pool was {len(w_treated_full)})")
print(f"IPTW control pool restricted to matched-control-eligible: {len(w_control_restricted)} "
      f"(full IPTW control pool was {len(w_control_full)})")

def weighted_rate(vals, w):
    return np.average(vals, weights=w)

rate_t_restricted = weighted_rate(
    w_treated_restricted["CONVERTED_POST_EXPOSURE"].values,
    w_treated_restricted["IPTW_WEIGHT_TRIMMED"].values,
)
rate_c_restricted = weighted_rate(
    w_control_restricted["CONVERTED_POST_EXPOSURE"].values,
    w_control_restricted["IPTW_WEIGHT_TRIMMED"].values,
)
lift_restricted = rate_t_restricted - rate_c_restricted

print(f"\n--- IPTW restricted to matched population ---")
print(f"Treated weighted conv rate: {rate_t_restricted:.5f}")
print(f"Control weighted conv rate: {rate_c_restricted:.5f}")
print(f"Lift: {lift_restricted:.5f} ({100*lift_restricted/rate_c_restricted:.2f}% relative)")

# ------------------------------------------------------------------
# (b) For direct comparison: full-sample IPTW (as originally reported)
# ------------------------------------------------------------------
rate_t_full = weighted_rate(w_treated_full["CONVERTED_POST_EXPOSURE"].values, w_treated_full["IPTW_WEIGHT_TRIMMED"].values)
rate_c_full = weighted_rate(w_control_full["CONVERTED_POST_EXPOSURE"].values, w_control_full["IPTW_WEIGHT_TRIMMED"].values)
lift_full = rate_t_full - rate_c_full
print(f"\n--- IPTW full sample (original) ---")
print(f"Treated weighted conv rate: {rate_t_full:.5f}")
print(f"Control weighted conv rate: {rate_c_full:.5f}")
print(f"Lift: {lift_full:.5f} ({100*lift_full/rate_c_full:.2f}% relative)")

# ------------------------------------------------------------------
# (c) Matched estimate (original, for 3-way comparison)
# ------------------------------------------------------------------
rate_t_matched = m_treated["CONVERTED_POST_EXPOSURE"].mean()
rate_c_matched = m_control["CONVERTED_POST_EXPOSURE"].mean()
lift_matched = rate_t_matched - rate_c_matched
print(f"\n--- Matched (original) ---")
print(f"Treated conv rate: {rate_t_matched:.5f}")
print(f"Control conv rate: {rate_c_matched:.5f}")
print(f"Lift: {lift_matched:.5f} ({100*lift_matched/rate_c_matched:.2f}% relative)")

# ------------------------------------------------------------------
# (d) Compare propensity score distributions: matched-treated vs
#     full-treated, and matched-control-pool vs full-control-pool
# ------------------------------------------------------------------
print("\n--- Propensity score distribution comparison ---")
print("Full treated pool propensity:", w_treated_full["PROPENSITY_SCORE"].describe()[["mean", "50%", "std"]].to_dict())
print("Matched treated propensity:  ", w_treated_restricted["PROPENSITY_SCORE"].describe()[["mean", "50%", "std"]].to_dict())
print("Full control pool propensity:", w_control_full["PROPENSITY_SCORE"].describe()[["mean", "50%", "std"]].to_dict())
print("Matched-eligible control propensity:", w_control_restricted["PROPENSITY_SCORE"].describe()[["mean", "50%", "std"]].to_dict())

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
summary = pd.DataFrame({
    "method": ["matched (original)", "IPTW full sample (original)", "IPTW restricted to matched population"],
    "rel_lift_pct": [
        100 * lift_matched / rate_c_matched,
        100 * lift_full / rate_c_full,
        100 * lift_restricted / rate_c_restricted,
    ],
})
print(summary.to_string(index=False))
