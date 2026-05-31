
-- ============================================
-- 1. CLV and probability alive by segment
-- ============================================

SELECT clv_segment,
       COUNT(*) AS customer_count,
       ROUND(AVG(clv_12m), 2) AS avg_clv,
       ROUND(SUM(clv_12m), 0) AS total_clv,
       ROUND(AVG(prob_alive) * 100, 1) AS avg_prob_alive_pct,
       ROUND(AVG(frequency), 1) AS avg_repeat_purchases
FROM customer_clv
GROUP BY clv_segment
ORDER BY avg_clv DESC;


-- ============================================
-- 2. At-risk high-value customers
-- ============================================

SELECT customer_id,
       clv_12m,
       prob_alive,
       frequency,
       recency,
       monetary_value

FROM customer_clv

WHERE clv_segment IN (
    'High Value',
    'Champions'
)

AND prob_alive < 0.5

ORDER BY clv_12m DESC

LIMIT 20;


-- ============================================
-- 3. Revenue concentration
-- ============================================

SELECT clv_segment,

ROUND(
    SUM(clv_12m) * 100.0 /

    (SELECT SUM(clv_12m)
     FROM customer_clv),

1) AS clv_share_pct

FROM customer_clv

GROUP BY clv_segment;


-- ============================================
-- 4. Dormant high-frequency customers
-- ============================================

SELECT customer_id,
       frequency,
       recency,
       T,
       prob_alive,
       clv_12m

FROM customer_clv

WHERE frequency > 5

AND prob_alive < 0.3

ORDER BY frequency DESC

LIMIT 20;
