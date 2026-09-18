"""
Step 16: Balance diagnostics on the MERGED Apple dataset.

Computes Standardized Mean Difference (SMD) for each covariate (extended
to include the new S3-sourced features):
  (a) raw (unadjusted) test vs. control
  (b) after 1:1 caliper matching
  (c) after IPTW (stabilized, trimmed)

Target: SMD < 0.1 is considered well-balanced.
"""
import numpy as np
import pandas as pd

RAW_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_propensity_scored.parquet"
MATCHED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_matched_pairs.parquet"
WEIGHTED_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_iptw_weighted.parquet"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/merged_balance_table.csv"

covariates = [
    # original features
    "APP_COUNT", "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES",
    "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT", "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT",
    "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT", "LIGHT_APP_COUNT",
    "DAYS_SINCE_LAST_ACTIVE",
    "HIST_IMPRESSIONS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_ACTIVE_DAYS_7D",
    "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS", "CHANNEL_TILE_RATIO",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "STV_YEAR", "DMA_CODE",
    "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D", "TOTAL_APP_TIME_7D",
    "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
    # new S3-sourced features
    "USER_HI2A", "G_USER_HI2A", "G_USER_VCR", "G_USER_MEDIA_PRICE",
    "G_USER_VIDEO_START_SUM", "G_USER_DISCREPANCY_PERCENT", "G_USER_IMPRESSION_21D",
    "ACR_APP_AVG_USING_TIME", "ACR_APP_USED_APP_CNT", "ACR_APP_EDUCATION",
    "ACR_APP_LIFESTYLE", "ACR_APP_INFORMATION", "ACR_APP_VIDEOS", "ACR_APP_SPORTS",
    "AD_COUNT", "AVG_AD_LENGTH", "AD_CATEGORY_COUNT",
    "AVG_DAILY_HOURS_WATCHED", "AVG_DAILY_APP_OPENS", "AVG_SESSION_TIME", "NUM_UNIQUE_APPS",
    "RATIO_TIME_INFORMATION", "RATIO_TIME_EDUCATION", "RATIO_TIME_GAME", "RATIO_TIME_VIDEOS",
    "RATIO_TIME_SPORTS", "RATIO_TIME_LIFESTYLE",
    "RATIO_TIME_ENGLISH", "RATIO_TIME_SPANISH",
]


def smd(treated_vals, control_vals, weights_t=None, weights_c=None):
    if weights_t is None:
        mean_t = treated_vals.mean()
        var_t = treated_vals.var()
    else:
        mean_t = np.average(treated_vals, weights=weights_t)
        var_t = np.average((treated_vals - mean_t) ** 2, weights=weights_t)

    if weights_c is None:
        mean_c = control_vals.mean()
        var_c = control_vals.var()
    else:
        mean_c = np.average(control_vals, weights=weights_c)
        var_c = np.average((control_vals - mean_c) ** 2, weights=weights_c)

    pooled_std = np.sqrt((var_t + var_c) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_t - mean_c) / pooled_std


raw = pd.read_parquet(RAW_PATH)
raw_treated = raw[raw.EXPOSED == 1]
raw_control = raw[raw.EXPOSED == 0]

rows = []
for col in covariates:
    rows.append({
        "covariate": col,
        "smd_raw": smd(raw_treated[col].values, raw_control[col].values),
    })
balance = pd.DataFrame(rows).set_index("covariate")

matched = pd.read_parquet(MATCHED_PATH)
m_treated = matched[matched.GROUP == "treated"]
m_control = matched[matched.GROUP == "control"]

smd_matched = []
for col in covariates:
    smd_matched.append(smd(m_treated[col].values, m_control[col].values))
balance["smd_matched"] = smd_matched

weighted = pd.read_parquet(WEIGHTED_PATH)
w_treated = weighted[weighted.EXPOSED == 1]
w_control = weighted[weighted.EXPOSED == 0]

smd_iptw = []
for col in covariates:
    smd_iptw.append(
        smd(
            w_treated[col].values, w_control[col].values,
            weights_t=w_treated["IPTW_WEIGHT_TRIMMED"].values,
            weights_c=w_control["IPTW_WEIGHT_TRIMMED"].values,
        )
    )
balance["smd_iptw"] = smd_iptw

balance["abs_smd_raw"] = balance["smd_raw"].abs()
balance["abs_smd_matched"] = balance["smd_matched"].abs()
balance["abs_smd_iptw"] = balance["smd_iptw"].abs()
balance = balance.sort_values("abs_smd_raw", ascending=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 150)
print("Balance table (Standardized Mean Difference; target < 0.1 after adjustment):\n")
print(balance[["smd_raw", "smd_matched", "smd_iptw"]].round(4))

print("\n--- Summary ---")
print(f"Covariates with |SMD| > 0.1 RAW (unadjusted):     {(balance['abs_smd_raw'] > 0.1).sum()} / {len(covariates)}")
print(f"Covariates with |SMD| > 0.1 after MATCHING:       {(balance['abs_smd_matched'] > 0.1).sum()} / {len(covariates)}")
print(f"Covariates with |SMD| > 0.1 after IPTW:            {(balance['abs_smd_iptw'] > 0.1).sum()} / {len(covariates)}")
print(f"\nMax |SMD| raw:      {balance['abs_smd_raw'].max():.4f} ({balance['abs_smd_raw'].idxmax()})")
print(f"Max |SMD| matched:  {balance['abs_smd_matched'].max():.4f} ({balance['abs_smd_matched'].idxmax()})")
print(f"Max |SMD| iptw:     {balance['abs_smd_iptw'].max():.4f} ({balance['abs_smd_iptw'].idxmax()})")

balance.to_csv(OUT_PATH)
print(f"\nSaved balance table to {OUT_PATH}")
