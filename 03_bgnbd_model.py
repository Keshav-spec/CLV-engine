# 03_bgnbd_model.py
# BG/NBD Customer Purchase Prediction Model

import pandas as pd
import dill
import os
import matplotlib.pyplot as plt

from lifetimes import BetaGeoFitter

from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions
)

 # Create directories
 
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

print("=" * 60)
print("LOADING RFM SUMMARY DATA")
print("=" * 60)

 # Load RFM summary table
 
summary = pd.read_csv(
    "data/features/rfm_summary.csv",
    index_col=0
)

print(f"\nSummary Shape: {summary.shape}")

print("\nColumns:")
print(summary.columns.tolist())

print("\nFirst 5 Rows:")
print(summary.head())

 # Basic sanity checks

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

print("\nFrequency Statistics:")
print(summary["frequency"].describe())

print("\nRecency Statistics:")
print(summary["recency"].describe())

print("\nT Statistics:")
print(summary["T"].describe())

# Fit BG/NBD Model

print("\n" + "=" * 60)
print("TRAINING BG/NBD MODEL")
print("=" * 60)

# Penalizer helps avoid overfitting
bgf = BetaGeoFitter(
    penalizer_coef=0.001
)

bgf.fit(
    summary["frequency"],
    summary["recency"],
    summary["T"]
)

print("\nModel fitted successfully.")

# Model Parameters

print("\nModel Parameters:")
print(bgf.params_)

print("\nInterpretation:")

print("""
r, alpha:
    Control purchase rate heterogeneity

a, b:
    Control customer dropout behavior
""")

# Save Model

with open(
    "models/bgnbd_model.pkl",
    "wb"
) as f:

    dill.dump(bgf, f)

print("\nModel saved successfully.")

# Predict Future Purchases

print("\n" + "=" * 60)
print("PREDICTING FUTURE PURCHASES")
print("=" * 60)

# Predict expected purchases in next 12 weeks

summary["predicted_purchases_12w"] = (
    bgf.conditional_expected_number_of_purchases_up_to_time(

        12,

        summary["frequency"],

        summary["recency"],

        summary["T"]
    )
)

# Probability Customer Is Alive

summary["prob_alive"] = (
    bgf.conditional_probability_alive(

        summary["frequency"],

        summary["recency"],

        summary["T"]
    )
)

print("\nPredictions generated successfully.")

# Top Customers by Predicted Purchases

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS")
print("=" * 60)

top_customers = (
    summary.nlargest(
        10,
        "predicted_purchases_12w"
    )[
        [
            "frequency",
            "recency",
            "T",
            "predicted_purchases_12w",
            "prob_alive"
        ]
    ]
)

print(top_customers)

# Save Predictions

summary.to_csv(
    "data/features/bgnbd_predictions.csv"
)

print("\nPrediction dataset saved.")

# Visualization 1
# Frequency-Recency Matrix

print("\n" + "=" * 60)
print("GENERATING VALIDATION PLOTS")
print("=" * 60)

fig = plt.figure(figsize=(12, 6))

plot_frequency_recency_matrix(bgf)

plt.title(
    "Expected Purchases in Next 12 Weeks"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/frequency_recency_matrix.png",
    dpi=150
)

plt.close()

print("\nSaved frequency-recency matrix.")

# Visualization 2
# Probability Alive Matrix

fig = plt.figure(figsize=(12, 6))

plot_probability_alive_matrix(bgf)

plt.title(
    "Probability Customer Is Still Active"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/probability_alive_matrix.png",
    dpi=150
)

plt.close()

print("Saved probability alive matrix.")

# Visualization 3
# Calibration Plot

fig = plt.figure(figsize=(10, 6))

plot_period_transactions(bgf)

plt.title(
    "Predicted vs Actual Transaction Frequency"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/calibration_check.png",
    dpi=150
)

plt.close()

print("Saved calibration chart.")

# Additional Diagnostics

print("\n" + "=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)

print("\nAverage Predicted Purchases:")
print(
    summary["predicted_purchases_12w"]
    .mean()
)

print("\nAverage Probability Alive:")
print(
    summary["prob_alive"]
    .mean()
)

print("\nCustomers With >80% Alive Probability:")
print(
    (summary["prob_alive"] > 0.8).sum()
)

print("\nCustomers With <20% Alive Probability:")
print(
    (summary["prob_alive"] < 0.2).sum()
)

print("\nHighest Predicted Purchases:")
print(
    summary["predicted_purchases_12w"]
    .max()
)

print("\nBG/NBD modeling completed successfully.")
