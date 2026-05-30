# 02_rfm_features.py
# BG/NBD RFM Feature Engineering

import pandas as pd
import matplotlib.pyplot as plt
import os

from lifetimes.utils import (
    summary_data_from_transaction_data
)

# Create required directories

os.makedirs("data/features", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

print("=" * 60)
print("LOADING CLEANED TRANSACTION DATA")
print("=" * 60)

# Load cleaned transaction dataset

df = pd.read_csv(
    "data\\processed\\transactions_clean.csv",
    parse_dates=["invoicedate"]
)

print(f"\nDataset Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

# Basic inspection

print("\n" + "=" * 60)
print("DATA INSPECTION")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDate Range:")
print(
    df["invoicedate"].min(),
    "to",
    df["invoicedate"].max()
)

print("\nUnique Customers:")
print(df["customer_id"].nunique())

# Analysis Date
# One day after final transaction

analysis_date = (
    df["invoicedate"].max()
    + pd.Timedelta(days=1)
)

print("\nAnalysis Date:", analysis_date)

# Build BG/NBD Summary Table

print("\n" + "=" * 60)
print("BUILDING RFM SUMMARY TABLE")
print("=" * 60)

summary = summary_data_from_transaction_data(

    transactions=df,

    customer_id_col="customer_id",

    datetime_col="invoicedate",

    monetary_value_col="revenue",

    observation_period_end=analysis_date,

    freq="W"  # Weekly units
)

# Summary table inspection

print("\nSummary Shape:", summary.shape)

print("\nFirst 10 Rows:")
print(summary.head(10))

print("\nDescriptive Statistics:")
print(summary.describe())

# BG/NBD Metric Explanation

print("\n" + "=" * 60)
print("BG/NBD METRIC DEFINITIONS")
print("=" * 60)

print("""
frequency:
    Number of repeat purchases
    (first purchase excluded)

recency:
    Time between first and last purchase
    measured in weeks

T:
    Customer lifetime in dataset
    (first purchase until analysis date)

monetary_value:
    Average revenue per transaction
""")

# Filter Repeat Customers
# BG/NBD and Gamma-Gamma work better
# for repeat purchasers

summary_repeat = summary[
    summary["frequency"] > 0
]

print("\n" + "=" * 60)
print("REPEAT CUSTOMER FILTER")
print("=" * 60)

print(
    f"\nRepeat Customers: "
    f"{len(summary_repeat):,}"
)

print(
    f"Total Customers: "
    f"{len(summary):,}"
)

repeat_ratio = (
    len(summary_repeat) / len(summary)
) * 100

print(f"Repeat Ratio: {repeat_ratio:.2f}%")

# Save summary tables

summary.to_csv(
    "data/features/rfm_summary.csv"
)

summary_repeat.to_csv(
    "data/features/rfm_repeat.csv"
)

print("\nRFM summary tables saved successfully.")

# Visual Sanity Checks

print("\n" + "=" * 60)
print("GENERATING DISTRIBUTION PLOTS")
print("=" * 60)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

# --------------------------------------------
# Frequency Distribution
# --------------------------------------------

summary["frequency"].hist(
    ax=axes[0],
    bins=50
)

axes[0].set_title(
    "Frequency Distribution"
)

axes[0].set_xlabel(
    "Repeat Purchases"
)

axes[0].set_ylabel(
    "Customer Count"
)

# --------------------------------------------
# Recency Distribution
# --------------------------------------------

summary["recency"].hist(
    ax=axes[1],
    bins=50,
    color="orange"
)

axes[1].set_title(
    "Recency Distribution"
)

axes[1].set_xlabel(
    "Recency (Weeks)"
)

# --------------------------------------------
# Monetary Value Distribution
# --------------------------------------------

summary["monetary_value"].hist(
    ax=axes[2],
    bins=50,
    color="green"
)

axes[2].set_title(
    "Monetary Value Distribution"
)

axes[2].set_xlabel(
    "Average Order Value"
)

# --------------------------------------------
# Save plot
# --------------------------------------------

plt.tight_layout()

plot_path = (
    "outputs/plots/"
    "rfm_distributions.png"
)

plt.savefig(
    plot_path,
    dpi=150
)

plt.close()

print(f"\nSaved Plot:\n{plot_path}")

# Additional Sanity Checks

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

print("\nCustomers with Frequency = 0:")
print(
    (summary["frequency"] == 0).sum()
)

print("\nCustomers with Frequency > 10:")
print(
    (summary["frequency"] > 10).sum()
)

print("\nMaximum Frequency:")
print(
    summary["frequency"].max()
)

print("\nMaximum Monetary Value:")
print(
    summary["monetary_value"].max()
)

print("\nTop 10 High-Value Repeat Customers:")

top_customers = (
    summary_repeat
    .sort_values(
        by=["frequency", "monetary_value"],
        ascending=False
    )
    .head(10)
)

print(top_customers)

print("\nRFM feature engineering completed successfully.")
