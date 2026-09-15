"""
Step 1: Load raw CSV export from Snowflake, clean types, and save as Parquet
for fast reuse in subsequent modeling steps.
"""
import glob
import numpy as np
import pandas as pd

RAW_DIR = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/raw"
OUT_PATH = "/Users/dongd1.xu/Documents/hermes_agent/measurement/data/propensity_features_apple.parquet"

files = sorted(glob.glob(f"{RAW_DIR}/propensity_features_apple_export_*.csv.gz"))
print(f"Found {len(files)} files")

dtype_map = {
    "PSID": "string",  # kept as string -- see uniqueness note below; Snowflake NUMBER(38,0) exceeds int64 range for some values
    "EXPOSED": "int8",
    "CAMPAIGN_IMPRESSIONS": "int32",
    "CONVERTED_POST_EXPOSURE": "int8",
    "APP_COUNT": "int32",
    "TOTAL_ACTIVE_DAYS": "float64",
    "TOTAL_MINUTES": "float64",
    "ACTIVE_APP_COUNT": "int32",
    "DORMANT_APP_COUNT": "int32",
    "SLEEPING_APP_COUNT": "int32",
    "AT_RISK_APP_COUNT": "int32",
    "CHAMPION_APP_COUNT": "int32",
    "REGULAR_APP_COUNT": "int32",
    "LIGHT_APP_COUNT": "int32",
    "DAYS_SINCE_LAST_ACTIVE": "float64",
    "HIST_IMPRESSIONS_7D": "int32",
    "HIST_IMP_EVENTS_7D": "int32",
    "HIST_LOAD_EVENTS_7D": "int32",
    "HIST_CLICK_EVENTS_7D": "int32",
    "HIST_TRACKER_EVENTS_7D": "int32",
    "HIST_CAMPAIGNS_7D": "int32",
    "HIST_CTV_IMPRESSIONS_7D": "int32",
    "HIST_TILE_IMPRESSIONS_7D": "int32",
    "HIST_ACTIVE_DAYS_7D": "int32",
    "HIST_CLICK_RATE": "float64",
    "AVG_DAILY_IMPRESSIONS": "float64",
    "CHANNEL_TILE_RATIO": "float64",
    "HIST_MOBILE_CAMPAIGN_IMP_7D": "int32",
    "HIST_WEB_CAMPAIGN_IMP_7D": "int32",
    "HIST_TV_CAMPAIGN_IMP_7D": "int32",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D": "int32",
    "HIST_WEB_CAMPAIGN_COUNT_7D": "int32",
    "HIST_TV_CAMPAIGN_COUNT_7D": "int32",
    "HIST_MOBILE_CAMPAIGN_RATIO": "float64",
    "HIST_WEB_CAMPAIGN_RATIO": "float64",
    "HIST_TV_CAMPAIGN_RATIO": "float64",
    "DEVICE_COUNTRY": "object",
    "STV_YEAR": "float64",
    "FIRMWARE_CODE": "object",
    "SAMSUNG_AFFINITY": "object",
    "DMA_CODE": "float64",
    "DEVICE_LANGUAGE": "object",
    "MODEL_CODE": "object",
    "TOTAL_TV_USAGE_TIME_7D": "float64",
    "TOTAL_PROG_TIME_7D": "float64",
    "TOTAL_APP_TIME_7D": "float64",
    "TOTAL_HDMI_TIME_7D": "float64",
    "TOTAL_AD_TIME_7D": "float64",
    "TOTAL_VOD_TIME_7D": "float64",
}

cols = list(dtype_map.keys())

chunks = []
for i, f in enumerate(files):
    df = pd.read_csv(
        f,
        header=0,
        names=cols,
        na_values=["\\N"],
        compression="gzip",
        low_memory=False,
    )
    chunks.append(df)
    if (i + 1) % 16 == 0:
        print(f"Loaded {i+1}/{len(files)} files")

full = pd.concat(chunks, ignore_index=True)
del chunks
print("Total rows:", len(full))

# CRITICAL: PSID must be kept as a string, NEVER cast to int64 or float64.
# Two independent precision-loss failure modes were found empirically here:
#   1) pd.to_numeric(errors="coerce") on PSID round-trips through float64,
#      which only has 53 bits of mantissa -- this silently collapses many
#      distinct large PSIDs onto the same rounded value (e.g. many landed
#      on the exact INT64_MIN/INT64_MAX sentinel), corrupting downstream
#      matching/weighting by making millions of rows appear to share one ID.
#   2) Even a direct .astype("int64") raises OverflowError, because Snowflake
#      stores PSID as NUMBER(38,0) and several real PSID values exceed the
#      signed int64 range (max 9,223,372,036,854,775,807) -- e.g. one PSID
#      in this dataset is 15,193,463,195,813,936,410, which fits in
#      unsigned int64 but not signed int64.
# PSID is only ever used as a join/grouping key here, never in arithmetic,
# so keeping it as a string avoids both failure modes entirely.
full["PSID"] = full["PSID"].astype(str).str.strip()
assert full["PSID"].nunique() == len(full), (
    f"PSID is not unique after loading: {full['PSID'].nunique()} distinct / {len(full)} rows. "
    "This indicates a precision-loss bug in loading -- do not proceed."
)
print("PSID uniqueness check passed:", full["PSID"].nunique(), "distinct PSIDs")

# Cast remaining numeric columns, coercing errors to NaN then filling per-column defaults later
for col, dt in dtype_map.items():
    if col == "PSID":
        continue
    if dt in ("int8", "int32", "int64"):
        full[col] = pd.to_numeric(full[col], errors="coerce")
    elif dt == "float64":
        full[col] = pd.to_numeric(full[col], errors="coerce")

# Integer label/count columns: fill NaN with 0 then cast to int
int_fill0_cols = [
    "CAMPAIGN_IMPRESSIONS", "APP_COUNT", "ACTIVE_APP_COUNT", "DORMANT_APP_COUNT",
    "SLEEPING_APP_COUNT", "AT_RISK_APP_COUNT", "CHAMPION_APP_COUNT", "REGULAR_APP_COUNT",
    "LIGHT_APP_COUNT", "HIST_IMPRESSIONS_7D", "HIST_IMP_EVENTS_7D", "HIST_LOAD_EVENTS_7D",
    "HIST_CLICK_EVENTS_7D", "HIST_TRACKER_EVENTS_7D", "HIST_CAMPAIGNS_7D",
    "HIST_CTV_IMPRESSIONS_7D", "HIST_TILE_IMPRESSIONS_7D", "HIST_ACTIVE_DAYS_7D",
    "HIST_MOBILE_CAMPAIGN_IMP_7D", "HIST_WEB_CAMPAIGN_IMP_7D", "HIST_TV_CAMPAIGN_IMP_7D",
    "HIST_MOBILE_CAMPAIGN_COUNT_7D", "HIST_WEB_CAMPAIGN_COUNT_7D", "HIST_TV_CAMPAIGN_COUNT_7D",
]
for col in int_fill0_cols:
    full[col] = full[col].fillna(0).astype("int32")

full["EXPOSED"] = full["EXPOSED"].fillna(0).astype("int8")
full["CONVERTED_POST_EXPOSURE"] = full["CONVERTED_POST_EXPOSURE"].fillna(0).astype("int8")
# PSID already validated as a unique string identifier above -- no further cast needed.

float_fill0_cols = [
    "TOTAL_ACTIVE_DAYS", "TOTAL_MINUTES", "HIST_CLICK_RATE", "AVG_DAILY_IMPRESSIONS",
    "CHANNEL_TILE_RATIO", "HIST_MOBILE_CAMPAIGN_RATIO", "HIST_WEB_CAMPAIGN_RATIO",
    "HIST_TV_CAMPAIGN_RATIO", "TOTAL_TV_USAGE_TIME_7D", "TOTAL_PROG_TIME_7D",
    "TOTAL_APP_TIME_7D", "TOTAL_HDMI_TIME_7D", "TOTAL_AD_TIME_7D", "TOTAL_VOD_TIME_7D",
]
for col in float_fill0_cols:
    full[col] = full[col].fillna(0.0)

# DAYS_SINCE_LAST_ACTIVE and STV_YEAR / DMA_CODE: keep NaN as missing-value signal,
# add explicit missing-indicator flags before imputation (done in modeling step).
full["DAYS_SINCE_LAST_ACTIVE_MISSING"] = full["DAYS_SINCE_LAST_ACTIVE"].isna().astype("int8")
full["DEVICE_COUNTRY_MISSING"] = full["DEVICE_COUNTRY"].isna().astype("int8")

for col in ["DEVICE_COUNTRY", "FIRMWARE_CODE", "SAMSUNG_AFFINITY", "DEVICE_LANGUAGE", "MODEL_CODE"]:
    full[col] = full[col].fillna("UNKNOWN").astype("category")

print("Dtypes:")
print(full.dtypes)
print("\nShape:", full.shape)
print("\nExposed counts:")
print(full["EXPOSED"].value_counts())
print("\nConverted counts:")
print(full["CONVERTED_POST_EXPOSURE"].value_counts())

full.to_parquet(OUT_PATH, index=False, engine="pyarrow")
print(f"\nSaved to {OUT_PATH}")
