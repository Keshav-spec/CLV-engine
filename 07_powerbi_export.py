import pandas as pd
import sqlite3
import os

# Create Power BI export folder


os.makedirs("powerbi", exist_ok=True)

print("Connecting to SQLite database...")

# Connect database

conn = sqlite3.connect("data/clv.db")

print("Database connected successfully.")

# Export 1: Full CLV customer table

print("\nExporting full customer CLV table...")

pd.read_sql(
    "SELECT * FROM customer_clv",
    conn
).to_csv(
    "powerbi/clv_customers.csv",
    index=False
)

# Export 2: Segment summary

print("Exporting segment summary...")

pd.read_csv(
    "outputs/reports/segment_summary.csv"
).to_csv(
    "powerbi/segment_summary.csv",
    index=False
)

# Export 3: At-risk high-value customers

print("Exporting at-risk customers...")

pd.read_sql(
    """
    SELECT *
    FROM customer_clv

    WHERE clv_segment IN (
        'High Value',
        'Champions'
    )

    AND prob_alive < 0.5

    ORDER BY clv_12m DESC
    """,
    conn
).to_csv(
    "powerbi/at_risk_customers.csv",
    index=False
)

# Export 4: CLV distribution dataset

print("Exporting CLV distribution dataset...")

pd.read_sql(
    """
    SELECT
        clv_12m,
        clv_segment,
        prob_alive,
        frequency,
        monetary_value

    FROM customer_clv
    """,
    conn
).to_csv(
    "powerbi/clv_distribution.csv",
    index=False
)

# Close database connection

conn.close()

print("\nAll Power BI CSV files exported successfully.")

print("\nExported files:")

print("- clv_customers.csv")
print("- segment_summary.csv")
print("- at_risk_customers.csv")
print("- clv_distribution.csv")
