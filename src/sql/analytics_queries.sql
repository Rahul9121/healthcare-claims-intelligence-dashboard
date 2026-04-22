-- 1) Monthly utilization and spend trend
SELECT
    DATE_TRUNC('month', service_date)::date AS service_month,
    COUNT(*) AS total_claims,
    COUNT(DISTINCT member_id) AS unique_members,
    ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay,
    ROUND(AVG(readmitted_30d::numeric) * 100, 2) AS readmission_rate_pct,
    ROUND(SUM(paid_amount), 2) AS total_paid_amount
FROM claims
GROUP BY 1
ORDER BY 1;

-- 2) Cost drivers by diagnosis
SELECT
    diagnosis_group,
    COUNT(*) AS claim_count,
    ROUND(SUM(paid_amount), 2) AS total_paid_amount,
    ROUND(AVG(paid_amount), 2) AS avg_paid_amount,
    ROUND(AVG(readmitted_30d::numeric) * 100, 2) AS readmission_rate_pct
FROM claims
GROUP BY 1
ORDER BY total_paid_amount DESC;

-- 3) Provider performance scorecard (minimum 30 claims)
SELECT
    provider_id,
    provider_name,
    provider_specialty,
    COUNT(*) AS claim_count,
    ROUND(AVG(paid_amount), 2) AS avg_paid_amount,
    ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay,
    ROUND(AVG(followup_within_7d::numeric) * 100, 2) AS followup_rate_pct,
    ROUND(AVG(readmitted_30d::numeric) * 100, 2) AS readmission_rate_pct
FROM claims
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 30
ORDER BY readmission_rate_pct ASC, avg_paid_amount DESC;

-- 4) Cohort analysis: plan x region x chronic burden
WITH member_level AS (
    SELECT
        m.member_id,
        m.plan_type,
        m.region,
        CASE
            WHEN m.chronic_conditions_count <= 1 THEN '0-1'
            WHEN m.chronic_conditions_count <= 3 THEN '2-3'
            ELSE '4+'
        END AS chronic_band,
        SUM(c.paid_amount) AS total_paid,
        AVG(c.readmitted_30d::numeric) * 100 AS readmission_rate_pct
    FROM members m
    JOIN claims c ON c.member_id = m.member_id
    GROUP BY 1, 2, 3, 4
)
SELECT
    plan_type,
    region,
    chronic_band,
    COUNT(*) AS members,
    ROUND(SUM(total_paid), 2) AS cohort_total_paid,
    ROUND(AVG(total_paid), 2) AS avg_paid_per_member,
    ROUND(AVG(readmission_rate_pct), 2) AS avg_readmission_rate_pct
FROM member_level
GROUP BY 1, 2, 3
ORDER BY cohort_total_paid DESC;

-- 5) Readmission reduction opportunity: follow-up within 7 days
SELECT
    followup_within_7d,
    COUNT(*) AS claims,
    ROUND(AVG(readmitted_30d::numeric) * 100, 2) AS readmission_rate_pct,
    ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay
FROM claims
GROUP BY 1
ORDER BY 1 DESC;
