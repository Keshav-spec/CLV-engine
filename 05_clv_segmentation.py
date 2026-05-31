# CLV Segmentation & Business Analytics
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os

 # Create directories
 
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("sql", exist_ok=True)

print("=" * 60)
print("LOADING CLV DATA")
print("=" * 60)

 # Load CLV scored dataset
 
df = pd.read_csv(
    "data/features/clv_scored.csv",
    index_col=0
)

print(f"\nDataset Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

 # Create CLV Segments
 
print("\n" + "=" * 60)
print("CREATING CLV SEGMENTS")
print("=" * 60)

df["clv_segment"] = pd.qcut(
    df["clv_12m"],
    q=[0, 0.25, 0.50, 0.75, 1.0],
    labels=[
        "Low Value",
        "Medium Value",
        "High Value",
        "Champions"
    ]
)

print("\nSegment Distribution:")

print(
    df["clv_segment"]
    .value_counts()
)

 # Segment Summary Statistics
 
print("\n" + "=" * 60)
print("SEGMENT SUMMARY")
print("=" * 60)

segment_summary = (
    df.groupby("clv_segment")
    .agg(
        customer_count=("clv_12m", "count"),

        avg_clv=("clv_12m", "mean"),

        total_clv=("clv_12m", "sum"),

        avg_frequency=("frequency", "mean"),

        avg_prob_alive=("prob_alive", "mean"),

        avg_monetary=("monetary_value", "mean")
    )
    .round(2)
)

segment_summary["clv_share_pct"] = (

    segment_summary["total_clv"]

    / segment_summary["total_clv"].sum()

    * 100
).round(1)

print("\nSegment Summary:")
print(segment_summary)

 # Save Segment Summary
 
segment_summary.to_csv(
    "outputs/reports/segment_summary.csv"
)

print("\nSegment summary saved.")

 # Pareto Analysis
 
print("\n" + "=" * 60)
print("PARETO ANALYSIS")
print("=" * 60)

df_sorted = df.sort_values(
    "clv_12m",
    ascending=False
)

df_sorted["cumulative_clv_pct"] = (

    df_sorted["clv_12m"].cumsum()

    / df_sorted["clv_12m"].sum()

    * 100
)

top_20pct = int(len(df) * 0.2)

top_20_clv = (

    df_sorted.iloc[:top_20pct]
    ["cumulative_clv_pct"]
    .iloc[-1]
)

print(
    f"\nTop 20% customers contribute "
    f"{top_20_clv:.1f}% of total CLV"
)

 # Visualization 1
# Segment Counts
 
fig = plt.figure(figsize=(8, 5))

df["clv_segment"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Customer Count by CLV Segment"
)

plt.xlabel("CLV Segment")

plt.ylabel("Customer Count")

plt.tight_layout()

plt.savefig(
    "outputs/plots/clv_segment_counts.png",
    dpi=150
)

plt.close()

 # Visualization 2
# CLV Share by Segment
 
fig = plt.figure(figsize=(8, 5))

segment_summary["clv_share_pct"].plot(
    kind="bar"
)

plt.title(
    "CLV Contribution by Segment"
)

plt.xlabel("Segment")

plt.ylabel("CLV Share (%)")

plt.tight_layout()

plt.savefig(
    "outputs/plots/clv_share_by_segment.png",
    dpi=150
)

plt.close()

 # Visualization 3
# CLV Distribution
 
fig = plt.figure(figsize=(10, 6))

df["clv_12m"].hist(
    bins=50
)

plt.title(
    "Customer Lifetime Value Distribution"
)

plt.xlabel("CLV")

plt.ylabel("Customer Count")

plt.tight_layout()

plt.savefig(
    "outputs/plots/clv_distribution.png",
    dpi=150
)

plt.close()

print("\nVisualization plots saved.")

 # Identify At-Risk High-Value Customers
 
print("\n" + "=" * 60)
print("AT-RISK HIGH-VALUE CUSTOMERS")
print("=" * 60)

at_risk = df[
    (df["clv_segment"].isin([
        "High Value",
        "Champions"
    ]))
    &
    (df["prob_alive"] < 0.5)
]

print(
    f"\nAt-Risk High-Value Customers: "
    f"{len(at_risk)}"
)

print("\nTop 10 At-Risk Customers:")

print(
    at_risk.sort_values(
        by="clv_12m",
        ascending=False
    )[
        [
            "frequency",
            "recency",
            "prob_alive",
            "clv_12m"
        ]
    ].head(10)
)

 # Save At-Risk Customers
 
at_risk.to_csv(
    "outputs/reports/at_risk_customers.csv"
)

print("\nAt-risk customer report saved.")

 # SQLite Database Export
 
print("\n" + "=" * 60)
print("EXPORTING TO SQLITE")
print("=" * 60)

conn = sqlite3.connect(
    "data/clv.db"
)

df.to_sql(
    "customer_clv",
    conn,
    if_exists="replace",
    index=True
)

conn.close()

print("\nSQLite database created successfully.")

 # Create SQL Query File
 
sql_queries = """
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
"""

with open(
    "sql/analysis_queries.sql",
    "w"
) as f:

    f.write(sql_queries)

print("\nSQL query file saved.")

 # Final Diagnostics
 
print("\n" + "=" * 60)
print("FINAL DIAGNOSTICS")
print("=" * 60)

print("\nAverage CLV by Segment:")

print(
    df.groupby("clv_segment")
    ["clv_12m"]
    .mean()
)

print("\nHighest CLV Customer:")
print(df["clv_12m"].max())

print("\nAverage Probability Alive:")
print(df["prob_alive"].mean())

print("\nSegmentation pipeline completed successfully.")
