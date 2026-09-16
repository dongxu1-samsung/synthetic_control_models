USE WAREHOUSE UDW_PLATFORM_INTELLIGENCE_CS_WH_PROD_LARGE;  -- default warehouse has a 30-min statement timeout, too small for this query

SET PRE_CAMPAIGN_START = '2026-06-24';    -- 7-day pre-campaign feature window start
SET PRE_CAMPAIGN_END = '2026-06-30';      -- 7-day pre-campaign feature window end
SET CAMPAIGN_START = '2026-07-01';        -- Campaign observation window start (flight 648768)
SET CAMPAIGN_END = '2026-08-01';          -- Campaign observation window end (flight 648768)
SET MAPPING_DATE_START = '2026-06-03';    -- AB_ML_MAPPING date filter

-- Campaign: 291260 | Flight: 648768 | App: Hulu (app_id 3201601007625)
CREATE OR REPLACE TABLE UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.PROPENSITY_FEATURES_HULU AS (
    -- ============================================================
    -- Category 1: Device Activity (P0)
    -- Source: DIM_USER_APP_STATE_SCD2
    -- ============================================================
    -- NOTE: Point-in-time filter (VALID_FROM/VALID_TO) is used instead of
    -- IS_CURRENT = TRUE. IS_CURRENT reflects app-state as of query run-time,
    -- which for a flight that already ended would pull in-flight and
    -- post-flight behavior (potentially influenced by the ad exposure itself)
    -- into what is supposed to be a pre-period covariate. Snapshotting as of
    -- $PRE_CAMPAIGN_END avoids this leakage.
    WITH device_activity AS (
        SELECT
            TRY_CAST(PSID AS NUMBER(38,0)) AS psid,
            COUNT(DISTINCT APP_ID) AS app_count,
            COALESCE(SUM(TOTAL_ACTIVE_DAYS), 0) AS total_active_days,
            COALESCE(SUM(TOTAL_MINUTES), 0) AS total_minutes,
            SUM(CASE WHEN LIFECYCLE_STAGE = 'Active' THEN 1 ELSE 0 END) AS active_app_count,
            SUM(CASE WHEN LIFECYCLE_STAGE = 'Dormant' THEN 1 ELSE 0 END) AS dormant_app_count,
            SUM(CASE WHEN LIFECYCLE_STAGE = 'Sleeping' THEN 1 ELSE 0 END) AS sleeping_app_count,
            SUM(CASE WHEN LIFECYCLE_STAGE = 'At_Risk' THEN 1 ELSE 0 END) AS at_risk_app_count,
            SUM(CASE WHEN USER_STATE = 'Champion' THEN 1 ELSE 0 END) AS champion_app_count,
            SUM(CASE WHEN USER_STATE = 'Regular' THEN 1 ELSE 0 END) AS regular_app_count,
            SUM(CASE WHEN USER_STATE = 'Light' THEN 1 ELSE 0 END) AS light_app_count,
            MAX(LAST_ACTIVE_DT) AS last_active_date
        FROM UDW_PROD.UDW_PLATFORM_INTELLIGENCE_CS.DIM_USER_APP_STATE_SCD2
        WHERE VALID_FROM <= $PRE_CAMPAIGN_END
          AND VALID_TO > $PRE_CAMPAIGN_END
        GROUP BY TRY_CAST(PSID AS NUMBER(38,0))
    ),

    -- ============================================================
    -- Category 2: Historical Ad Exposure (P1)
    -- Source: LOG_DELIVERY_RAW_WITHOUT_PII, 7-day pre-campaign period
    -- ============================================================
    hist_impressions AS (
        SELECT
            SAMSUNG_PSID_PII_VIRTUAL_ID AS psid,
            COUNT(DISTINCT BID_REQUEST_ID) AS hist_impressions_7d,
            COUNT(DISTINCT FLIGHT_ID) AS hist_campaigns_7d,
            COUNT(DISTINCT CASE WHEN EVENT = 'impression' THEN BID_REQUEST_ID END) AS hist_imp_events_7d,
            COUNT(DISTINCT CASE WHEN EVENT = 'load' THEN BID_REQUEST_ID END) AS hist_load_events_7d,
            COUNT(DISTINCT CASE WHEN EVENT = 'click' THEN BID_REQUEST_ID END) AS hist_click_events_7d,
            COUNT(DISTINCT CASE WHEN EVENT = 'tracker' THEN BID_REQUEST_ID END) AS hist_tracker_events_7d,
            COUNT(DISTINCT CASE WHEN CHANNEL = 'ctv' AND EVENT = 'impression' THEN BID_REQUEST_ID END) AS hist_ctv_impressions_7d,
            COUNT(DISTINCT CASE WHEN CHANNEL = 'tile' AND EVENT = 'impression' THEN BID_REQUEST_ID END) AS hist_tile_impressions_7d,
            COUNT(DISTINCT DAY) AS hist_active_days_7d
        FROM UDW_PROD.TRADER.LOG_DELIVERY_RAW_WITHOUT_PII
        WHERE UDW_PARTITION_DATETIME BETWEEN $PRE_CAMPAIGN_START AND $PRE_CAMPAIGN_END
          AND SAMSUNG_PSID_PII_VIRTUAL_ID IS NOT NULL
          AND EVENT IN ('impression', 'load', 'click', 'tracker')
        GROUP BY SAMSUNG_PSID_PII_VIRTUAL_ID
    ),

    -- ============================================================
    -- Category 2b: Historical Campaign-Type-Specific Exposure (P1)
    -- Source: CONV_UPDATE_UNION, 7-day pre-campaign period
    -- Distinguishes Mobile / Web / TV campaign targeting history
    -- ============================================================
    hist_campaign_type AS (
        SELECT
            m.SAMSUNG_PSID AS psid,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'Mobile' THEN cu.BID_REQUEST_ID END) AS hist_mobile_campaign_imp_7d,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'Web' THEN cu.BID_REQUEST_ID END) AS hist_web_campaign_imp_7d,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'TV' THEN cu.BID_REQUEST_ID END) AS hist_tv_campaign_imp_7d,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'Mobile' THEN cu.FLIGHT_ID END) AS hist_mobile_campaign_count_7d,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'Web' THEN cu.FLIGHT_ID END) AS hist_web_campaign_count_7d,
            COUNT(DISTINCT CASE WHEN cu.FLIGHT_CONVERSION_CHANNELS = 'TV' THEN cu.FLIGHT_ID END) AS hist_tv_campaign_count_7d
        FROM UDW_PROD.UDW_PLATFORM_INTELLIGENCE_CS.CONV_UPDATE_UNION cu
        JOIN UDW_PROD.UDW_PLATFORM_INTELLIGENCE_CS.AB_ML_MAPPING_PSID_TVID_IP_WITHOUT_PII m
            ON cu.SAMSUNG_TVID = m.SAMSUNG_TVID
        WHERE cu.DT BETWEEN $PRE_CAMPAIGN_START AND $PRE_CAMPAIGN_END
          AND cu.EVENT = 'impression'
          AND m.DATE >= $PRE_CAMPAIGN_START
        GROUP BY m.SAMSUNG_PSID
    ),

    -- ============================================================
    -- Category 3: Device Demographics (P2)
    -- Source: LOG_DELIVERY_RAW_WITHOUT_PII, 7-day pre-campaign period
    -- ============================================================
    device_demo AS (
        SELECT DISTINCT
            SAMSUNG_PSID_PII_VIRTUAL_ID AS psid,
            FIRST_VALUE(SAMSUNG_DEVICE_COUNTRY) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS device_country,
            FIRST_VALUE(STV_YEAR) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS stv_year,
            FIRST_VALUE(SAMSUNG_FIRMWARE_CODE) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS firmware_code,
            FIRST_VALUE(SAMSUNG_AFFINITY) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS samsung_affinity,
            FIRST_VALUE(DMA_CODE) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS dma_code,
            FIRST_VALUE(LANGUAGE) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS device_language,
            FIRST_VALUE(SAMSUNG_MODEL_CODE) OVER (
                PARTITION BY SAMSUNG_PSID_PII_VIRTUAL_ID
                ORDER BY UDW_PARTITION_DATETIME DESC
            ) AS model_code
        FROM UDW_PROD.TRADER.LOG_DELIVERY_RAW_WITHOUT_PII
        WHERE UDW_PARTITION_DATETIME BETWEEN $PRE_CAMPAIGN_START AND $PRE_CAMPAIGN_END
          AND SAMSUNG_PSID_PII_VIRTUAL_ID IS NOT NULL
          AND EVENT = 'impression'
    ),
    device_demo_dedup AS (
        SELECT psid,
            MAX(device_country) AS device_country,
            MAX(stv_year) AS stv_year,
            MAX(firmware_code) AS firmware_code,
            MAX(samsung_affinity) AS samsung_affinity,
            MAX(dma_code) AS dma_code,
            MAX(device_language) AS device_language,
            MAX(model_code) AS model_code
        FROM device_demo
        GROUP BY psid
    ),
    -- ============================================================
    -- Category 4: TV Usage
    -- Source: AUDIENCE_INTELLIGENCE_LL_EXP_DAILY_FRACTION_WITHOUT_PII, 7-day pre-campaign period
    -- ============================================================
    tv_usage AS (
        SELECT
            PSID AS psid,
            SUM(TOTAL_USAGE_TIME) AS total_tv_usage_time_7d,
            SUM(PROG) AS total_prog_time_7d,
            SUM(APP) AS total_app_time_7d,
            SUM(HDMI) AS total_hdmi_time_7d,
            SUM(AD) AS total_ad_time_7d,
            SUM(VOD) AS total_vod_time_7d,
            COUNT(DISTINCT PARTITION_DATE) AS tv_active_days_7d
        FROM UDW_PROD.UDW_PLATFORM_INTELLIGENCE_CS.AUDIENCE_INTELLIGENCE_LL_EXP_DAILY_FRACTION_WITHOUT_PII
        WHERE PARTITION_DATE BETWEEN '20260624' AND '20260630'
        GROUP BY PSID
    ),
    -- ============================================================
    -- Pre-computed eligible user pool along with exposure and conversion
    -- Source: SYNTHETIC_CONTROL_MEASUREMENT_USER_POOL_3201601007625 (flight 648768, Hulu / campaign 291260)
    --   Exposed (test)      : ELIGIBLE_BID_REQUESTS_WON_AUCTION_DELIVERED_IMPRESSION > 0
    --   Non-exposed (control): ELIGIBLE_BID_REQUESTS_ENTERED_AUCTION > 0
    --                          AND ELIGIBLE_BID_REQUESTS_WON_AUCTION_DELIVERED_IMPRESSION = 0
    --   Converted           : RAW_CONVERSIONS > 0  (NOTE: raw count, not CONV_PROB)
    -- ============================================================
    audience_users AS (
        SELECT
            PSID_PII_VIRTUAL_ID AS psid,
            CASE WHEN ELIGIBLE_BID_REQUESTS_WON_AUCTION_DELIVERED_IMPRESSION > 0 THEN 1 ELSE 0 END AS is_exposed,
            CASE WHEN RAW_CONVERSIONS > 0 THEN 1 ELSE 0 END AS is_converted_post_exposure,
            ELIGIBLE_BID_REQUESTS_WON_AUCTION_DELIVERED_IMPRESSION AS campaign_impressions
        FROM UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.SYNTHETIC_CONTROL_MEASUREMENT_USER_POOL_3201601007625
        WHERE ELIGIBLE_BID_REQUESTS_ENTERED_AUCTION > 0  -- restrict to the truly eligible (in-auction) population
    )

    -- ============================================================
    -- Final feature table
    -- ============================================================
    SELECT
        a.psid,
        -- Label
        COALESCE(a.is_exposed, 0) AS exposed,
        COALESCE(a.campaign_impressions, 0) AS campaign_impressions,
        COALESCE(a.is_converted_post_exposure, 0) AS converted_post_exposure,

        -- Category 1: Device Activity (P0)
        COALESCE(da.app_count, 0) AS app_count,
        COALESCE(da.total_active_days, 0) AS total_active_days,
        COALESCE(da.total_minutes, 0) AS total_minutes,
        COALESCE(da.active_app_count, 0) AS active_app_count,
        COALESCE(da.dormant_app_count, 0) AS dormant_app_count,
        COALESCE(da.sleeping_app_count, 0) AS sleeping_app_count,
        COALESCE(da.at_risk_app_count, 0) AS at_risk_app_count,
        COALESCE(da.champion_app_count, 0) AS champion_app_count,
        COALESCE(da.regular_app_count, 0) AS regular_app_count,
        COALESCE(da.light_app_count, 0) AS light_app_count,
        DATEDIFF('day', da.last_active_date, $PRE_CAMPAIGN_END) AS days_since_last_active,

        -- Category 2: Historical Ad Exposure (P1, 7-day window)
        COALESCE(hi.hist_impressions_7d, 0) AS hist_impressions_7d,
        COALESCE(hi.hist_imp_events_7d, 0) AS hist_imp_events_7d,
        COALESCE(hi.hist_load_events_7d, 0) AS hist_load_events_7d,
        COALESCE(hi.hist_click_events_7d, 0) AS hist_click_events_7d,
        COALESCE(hi.hist_tracker_events_7d, 0) AS hist_tracker_events_7d,
        COALESCE(hi.hist_campaigns_7d, 0) AS hist_campaigns_7d,
        COALESCE(hi.hist_ctv_impressions_7d, 0) AS hist_ctv_impressions_7d,
        COALESCE(hi.hist_tile_impressions_7d, 0) AS hist_tile_impressions_7d,
        COALESCE(hi.hist_active_days_7d, 0) AS hist_active_days_7d,
        CASE WHEN COALESCE(hi.hist_imp_events_7d, 0) > 0
             THEN hi.hist_click_events_7d::FLOAT / hi.hist_imp_events_7d
             ELSE 0 END AS hist_click_rate,
        CASE WHEN COALESCE(hi.hist_active_days_7d, 0) > 0
             THEN hi.hist_imp_events_7d::FLOAT / hi.hist_active_days_7d
             ELSE 0 END AS avg_daily_impressions,
        CASE WHEN COALESCE(hi.hist_imp_events_7d, 0) > 0
             THEN hi.hist_tile_impressions_7d::FLOAT / hi.hist_imp_events_7d
             ELSE 0 END AS channel_tile_ratio,

        -- Category 2b: Historical Campaign-Type-Specific Exposure (P1, 7-day window)
        COALESCE(hct.hist_mobile_campaign_imp_7d, 0) AS hist_mobile_campaign_imp_7d,
        COALESCE(hct.hist_web_campaign_imp_7d, 0) AS hist_web_campaign_imp_7d,
        COALESCE(hct.hist_tv_campaign_imp_7d, 0) AS hist_tv_campaign_imp_7d,
        COALESCE(hct.hist_mobile_campaign_count_7d, 0) AS hist_mobile_campaign_count_7d,
        COALESCE(hct.hist_web_campaign_count_7d, 0) AS hist_web_campaign_count_7d,
        COALESCE(hct.hist_tv_campaign_count_7d, 0) AS hist_tv_campaign_count_7d,
        CASE WHEN COALESCE(hi.hist_imp_events_7d, 0) > 0
             THEN COALESCE(hct.hist_mobile_campaign_imp_7d, 0)::FLOAT / hi.hist_imp_events_7d
             ELSE 0 END AS hist_mobile_campaign_ratio,
        CASE WHEN COALESCE(hi.hist_imp_events_7d, 0) > 0
             THEN COALESCE(hct.hist_web_campaign_imp_7d, 0)::FLOAT / hi.hist_imp_events_7d
             ELSE 0 END AS hist_web_campaign_ratio,
        CASE WHEN COALESCE(hi.hist_imp_events_7d, 0) > 0
             THEN COALESCE(hct.hist_tv_campaign_imp_7d, 0)::FLOAT / hi.hist_imp_events_7d
             ELSE 0 END AS hist_tv_campaign_ratio,

        -- Category 3: Device Demographics (P2)
        dd.device_country,
        dd.stv_year,
        dd.firmware_code,
        dd.samsung_affinity,
        dd.dma_code,
        dd.device_language,
        dd.model_code,

        -- Category 4: TV Usage (P2)
        COALESCE(tv.total_tv_usage_time_7d, 0) AS total_tv_usage_time_7d,
        COALESCE(tv.total_prog_time_7d, 0) AS total_prog_time_7d,
        COALESCE(tv.total_app_time_7d, 0) AS total_app_time_7d,
        COALESCE(tv.total_hdmi_time_7d, 0) AS total_hdmi_time_7d,
        COALESCE(tv.total_ad_time_7d, 0) AS total_ad_time_7d,
        COALESCE(tv.total_vod_time_7d, 0) AS total_vod_time_7d

    FROM audience_users a
    LEFT JOIN device_activity da ON a.psid = da.psid
    LEFT JOIN hist_impressions hi ON a.psid = hi.psid
    LEFT JOIN hist_campaign_type hct ON a.psid = hct.psid
    LEFT JOIN device_demo_dedup dd ON a.psid = dd.psid
    LEFT JOIN tv_usage tv ON a.psid = tv.psid
)
;

-- ============================================================
-- Execution verification (run 2026-09-16 against Snowflake):
--   Table: UDW_PROD.UDW_PLATFORM_INTELLIGENCE_SS.PROPENSITY_FEATURES_HULU
--   Total rows / distinct psid : 5,964,077
--   Exposed (test)             : 734,974
--   Control (eligible, unexposed): 5,229,103
--   Converted (RAW_CONVERSIONS > 0): 217,283
--   Matches source pool SYNTHETIC_CONTROL_MEASUREMENT_USER_POOL_3201601007625
--   restricted to ELIGIBLE_BID_REQUESTS_ENTERED_AUCTION > 0.
-- ============================================================
