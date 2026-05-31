import pandas as pd
import dill
import os

from lifetimes import GammaGammaFitter

# Create directories

os.makedirs("models", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

print("Loading RFM summary data...")

# Load RFM summary table

summary = pd.read_csv(
    "data/features/rfm_summary.csv",
    index_col=0
)

print(f"\nSummary Shape: {summary.shape}")

print("\nColumns:")
print(summary.columns.tolist())

# Load BG/NBD predictions

print("\nLoading BG/NBD predictions...")

bgnbd_predictions = pd.read_csv(
    "data/features/bgnbd_predictions.csv",
    index_col=0
)

# Merge BG/NBD outputs into summary table

summary["prob_alive"] = (
    bgnbd_predictions["prob_alive"]
)

summary["predicted_purchases_12w"] = (
    bgnbd_predictions["predicted_purchases_12w"]
)

print("\nBG/NBD predictions merged successfully.")

print("\nFirst 5 Rows:")
print(summary.head())

# Load BG/NBD model

print("\nLoading BG/NBD model...")

with open(
    "models/bgnbd_model.pkl",
    "rb"
) as f:

    bgf = dill.load(f)

print("BG/NBD model loaded successfully.")

# Prepare Gamma-Gamma dataset

print("\nPreparing Gamma-Gamma dataset...")

gg_df = summary[
    (summary["frequency"] > 0)
    &
    (summary["monetary_value"] > 0)
].copy()

print(f"\nEligible Customers: {len(gg_df):,}")

# Validate Gamma-Gamma assumption

print("\nChecking model assumptions...")

corr = gg_df["frequency"].corr(
    gg_df["monetary_value"]
)

print(
    f"\nFrequency-Monetary Correlation: "
    f"{corr:.4f}"
)

print(
    "\nExpected correlation should "
    "be below 0.30"
)

if corr < 0.30:
    print("\nGamma-Gamma assumption satisfied.")
else:
    print("\nWARNING:")
    print(
        "High correlation detected."
    )
    print(
        "Results may be less reliable."
    )

# Fit Gamma-Gamma model

print("\nTraining Gamma-Gamma model...")

ggf = GammaGammaFitter(
    penalizer_coef=0.001
)

ggf.fit(
    gg_df["frequency"],
    gg_df["monetary_value"]
)

print("\nModel fitted successfully.")

# Display parameters

print("\nGamma-Gamma Parameters:")
print(ggf.params_)

# Save model

with open(
    "models/gamma_gamma_model.pkl",
    "wb"
) as f:

    dill.dump(ggf, f)

print("\nGamma-Gamma model saved.")

# Predict expected average profit

print("\nPredicting expected average order value...")

gg_df["expected_avg_profit"] = (
    ggf.conditional_expected_average_profit(

        gg_df["frequency"],

        gg_df["monetary_value"]
    )
)

print("Prediction completed.")

# Compute CLV

print("\nComputing customer lifetime value...")

clv = ggf.customer_lifetime_value(

    transaction_prediction_model=bgf,

    frequency=gg_df["frequency"],

    recency=gg_df["recency"],

    T=gg_df["T"],

    monetary_value=gg_df["monetary_value"],

    time=12,

    discount_rate=0.01
)

gg_df["clv_12m"] = clv

print("\nCLV computation completed.")

# CLV statistics

print("\nCLV Summary Statistics:")

print(
    gg_df["clv_12m"].describe()
)

# Top customers

print("\nTop 10 Customers by CLV:")

top_clv = (
    gg_df
    .sort_values(
        by="clv_12m",
        ascending=False
    )
    .head(10)
)

print(
    top_clv[
        [
            "frequency",
            "recency",
            "T",
            "monetary_value",
            "prob_alive",
            "predicted_purchases_12w",
            "expected_avg_profit",
            "clv_12m"
        ]
    ]
)

# Create preliminary segments

print("\nCreating preliminary CLV segments...")

gg_df["clv_segment"] = pd.qcut(
    gg_df["clv_12m"],
    q=4,
    labels=[
        "Low Value",
        "Medium Value",
        "High Value",
        "Champions"
    ]
)

print("\nSegment Distribution:")

print(
    gg_df["clv_segment"]
    .value_counts()
)

# Save final dataset

output_path = (
    "data/features/clv_scored.csv"
)

gg_df.to_csv(output_path)

print("\nCLV dataset saved successfully.")

print(f"\nSaved To:\n{output_path}")

# Diagnostics

print("\nDiagnostics:")

print(
    f"\nAverage CLV: "
    f"{gg_df['clv_12m'].mean():.2f}"
)

print(
    f"Median CLV: "
    f"{gg_df['clv_12m'].median():.2f}"
)

print(
    f"Highest CLV: "
    f"{gg_df['clv_12m'].max():.2f}"
)

print(
    f"Lowest CLV: "
    f"{gg_df['clv_12m'].min():.2f}"
)

print(
    f"\nCustomers with CLV > 1000: "
    f"{(gg_df['clv_12m'] > 1000).sum()}"
)

print("\nGamma-Gamma modeling completed successfully.")
