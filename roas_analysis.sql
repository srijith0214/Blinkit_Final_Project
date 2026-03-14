-- ============================================================
-- Blinkit Master Analytical View - ROAS Analysis
-- Author: Blinkit BI Platform
-- Description: Joins transactional orders data (granular) with
--   daily marketing spend data to solve the granularity mismatch
--   and calculate ROAS per day. Uses CTEs for readability.
-- ============================================================

-- Step 1: Load raw tables (assumes PostgreSQL with tables loaded)
-- CREATE TABLE IF NOT EXISTS orders (...);
-- CREATE TABLE IF NOT EXISTS marketing_performance (...);
-- COPY orders FROM 'blinkit_orders.csv' CSV HEADER;
-- COPY marketing_performance FROM 'blinkit_marketing_performance.csv' CSV HEADER;

-- ============================================================
-- MASTER ANALYTICAL VIEW
-- ============================================================
WITH

-- CTE 1: Squash thousands of orders into 1 row per day
daily_revenue AS (
    SELECT
        DATE(order_date)                        AS order_date,
        COUNT(order_id)                         AS total_orders,
        SUM(order_total)                        AS total_revenue,
        AVG(order_total)                        AS avg_order_value,
        SUM(CASE WHEN delivery_status != 'On Time' THEN 1 ELSE 0 END) AS delayed_orders,
        ROUND(
            100.0 * SUM(CASE WHEN delivery_status = 'On Time' THEN 1 ELSE 0 END)
            / COUNT(order_id), 2
        )                                       AS on_time_pct
    FROM orders
    GROUP BY DATE(order_date)
),

-- CTE 2: Aggregate marketing spend per day (already daily, but may have multiple channels)
daily_marketing AS (
    SELECT
        DATE(date)                              AS campaign_date,
        SUM(spend)                              AS total_spend,
        SUM(impressions)                        AS total_impressions,
        SUM(clicks)                             AS total_clicks,
        SUM(conversions)                        AS total_conversions,
        SUM(revenue_generated)                  AS marketing_attributed_revenue
    FROM marketing_performance
    GROUP BY DATE(date)
),

-- CTE 3: Join on date - solve the granularity mismatch
master_join AS (
    SELECT
        COALESCE(r.order_date, m.campaign_date) AS date,
        COALESCE(r.total_orders, 0)             AS total_orders,
        COALESCE(r.total_revenue, 0)            AS total_revenue,
        COALESCE(r.avg_order_value, 0)          AS avg_order_value,
        COALESCE(r.delayed_orders, 0)           AS delayed_orders,
        COALESCE(r.on_time_pct, 0)              AS on_time_pct,
        COALESCE(m.total_spend, 0)              AS total_spend,
        COALESCE(m.total_impressions, 0)        AS total_impressions,
        COALESCE(m.total_clicks, 0)             AS total_clicks,
        COALESCE(m.total_conversions, 0)        AS total_conversions,
        COALESCE(m.marketing_attributed_revenue, 0) AS marketing_attributed_revenue
    FROM daily_revenue r
    FULL OUTER JOIN daily_marketing m
        ON r.order_date = m.campaign_date
)

-- CTE 4: Calculate ROAS and label profitable days
SELECT
    date,
    total_orders,
    ROUND(total_revenue::NUMERIC, 2)            AS total_revenue,
    ROUND(avg_order_value::NUMERIC, 2)          AS avg_order_value,
    delayed_orders,
    on_time_pct,
    ROUND(total_spend::NUMERIC, 2)              AS total_spend,
    total_impressions,
    total_clicks,
    total_conversions,

    -- ROAS: Revenue / Spend (handle divide-by-zero for days with no spend)
    CASE
        WHEN total_spend = 0 THEN NULL          -- Organic day, no ad spend
        ELSE ROUND((total_revenue / total_spend)::NUMERIC, 4)
    END                                         AS roas,

    -- Marketing attribution gap
    ROUND(marketing_attributed_revenue::NUMERIC, 2) AS marketing_attributed_revenue,

    -- Business classification
    CASE
        WHEN total_spend = 0 THEN 'Organic'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 2.0 THEN 'Profitable'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 1.0 THEN 'Break-Even'
        ELSE 'Loss-Making'
    END                                         AS day_classification,

    -- CTR
    CASE
        WHEN total_impressions = 0 THEN 0
        ELSE ROUND((100.0 * total_clicks / total_impressions)::NUMERIC, 4)
    END                                         AS ctr_pct

FROM master_join
ORDER BY date;


-- ============================================================
-- CHANNEL ROAS ANALYSIS
-- ============================================================
SELECT
    channel,
    COUNT(DISTINCT campaign_id)                 AS num_campaigns,
    ROUND(SUM(spend)::NUMERIC, 2)               AS total_spend,
    ROUND(SUM(revenue_generated)::NUMERIC, 2)   AS total_revenue,
    ROUND((SUM(revenue_generated)/SUM(spend))::NUMERIC, 4) AS roas,
    SUM(impressions)                            AS total_impressions,
    SUM(conversions)                            AS total_conversions,
    ROUND((100.0*SUM(conversions)/SUM(impressions))::NUMERIC, 4) AS conversion_rate_pct
FROM marketing_performance
GROUP BY channel
ORDER BY roas DESC;


-- ============================================================
-- DELIVERY DELAY ANALYSIS
-- ============================================================
SELECT
    EXTRACT(HOUR FROM order_date::TIMESTAMP)    AS hour_of_day,
    COUNT(*)                                    AS total_orders,
    SUM(CASE WHEN delivery_status != 'On Time' THEN 1 ELSE 0 END) AS delayed,
    ROUND(
        100.0 * SUM(CASE WHEN delivery_status != 'On Time' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                           AS delay_rate_pct
FROM orders
GROUP BY EXTRACT(HOUR FROM order_date::TIMESTAMP)
ORDER BY hour_of_day;
