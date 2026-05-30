
 
# 01_data_cleaning.py
# Customer Lifetime Value (CLV) Project
 

import pandas as pd
import numpy as np
import os

 
# Create processed data directory
 

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

 
# Load Excel sheets
 

file_path = "data/raw/online_retail_II.csv" 
df = pd.read_csv(file_path) 
print(f"Initial Shape: {df.shape}")



 
# Standardize column names
 

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns:")
print(df.columns.tolist())

 
# Basic Dataset Inspection
 

print("\n" + "=" * 60)
print("INITIAL DATA INSPECTION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDate Range:")
print(
    df["invoicedate"].min(),
    "to",
    df["invoicedate"].max()
)

print("\nUnique Customers:")
print(df["customer_id"].nunique())

print("\nUnique Products:")
print(df["stockcode"].nunique())

print("\nUnique Countries:")
print(df["country"].nunique())

 
# Cleaning Function
 

def clean_transactions(df):

    print("\n" + "=" * 60)
    print("STARTING DATA CLEANING")
    print("=" * 60)

    initial_rows = len(df)

    
    # Remove duplicate rows
    

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows found: {duplicates}")

    df = df.drop_duplicates()

    print(f"Shape after duplicate removal: {df.shape}")

    
    # Remove missing customer IDs
    

    missing_customers = df["customer_id"].isnull().sum()

    print(f"\nMissing Customer IDs: {missing_customers}")

    df = df.dropna(subset=["customer_id"])

    print(f"Shape after removing missing customers: {df.shape}")

    
    # Remove cancelled invoices
    # Invoice starts with 'C'
    

    cancelled_orders = (
        df["invoice"]
        .astype(str)
        .str.startswith("C")
        .sum()
    )

    print(f"\nCancelled transactions removed: {cancelled_orders}")

    df = df[
        ~df["invoice"]
        .astype(str)
        .str.startswith("C")
    ]

    print(f"Shape after removing cancellations: {df.shape}")

    
    # Remove invalid quantities
    

    invalid_quantity = (df["quantity"] <= 0).sum()

    print(f"\nInvalid quantities removed: {invalid_quantity}")

    df = df[df["quantity"] > 0]

    print(f"Shape after quantity filtering: {df.shape}")

    
    # Remove invalid prices
    

    invalid_price = (df["price"] <= 0).sum()

    print(f"\nInvalid prices removed: {invalid_price}")

    df = df[df["price"] > 0]

    print(f"Shape after price filtering: {df.shape}")

    
    # Remove operational/non-product stock codes
    

    invalid_codes = [
        "POST",
        "D",
        "M",
        "BANK CHARGES",
        "PADS",
        "DOT"
    ]

    invalid_stock_rows = (
        df["stockcode"]
        .astype(str)
        .isin(invalid_codes)
        .sum()
    )

    print(f"\nOperational entries removed: {invalid_stock_rows}")

    df = df[
        ~df["stockcode"]
        .astype(str)
        .isin(invalid_codes)
    ]

    print(f"Shape after stock code cleaning: {df.shape}")

    
    # Create revenue column
    

    df["revenue"] = df["quantity"] * df["price"]

    
    # Convert datetime
    

    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    
    # Convert customer ID to integer
    

    df["customer_id"] = df["customer_id"].astype(int)

    
    # Final cleaning statistics
    

    final_rows = len(df)

    removed_rows = initial_rows - final_rows

    retention_rate = (final_rows / initial_rows) * 100

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"\nInitial Rows: {initial_rows}")
    print(f"Final Rows: {final_rows}")
    print(f"Rows Removed: {removed_rows}")

    print(f"\nRetention Rate: {retention_rate:.2f}%")

    print(f"\nCustomers Retained: {df['customer_id'].nunique()}")

    return df


 
# Apply cleaning
 

df = clean_transactions(df)

 
# Final Dataset Summary
 

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print(f"\nTransactions: {len(df):,}")

print(f"Customers: {df['customer_id'].nunique():,}")

print(f"Products: {df['stockcode'].nunique():,}")

print(f"Countries: {df['country'].nunique():,}")

print("\nRevenue Statistics:")
print(df["revenue"].describe())

print("\nTop 10 Countries by Revenue:")

country_revenue = (
    df.groupby("country")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(country_revenue)

 
# Save cleaned dataset
 

csv_output_path = "data/processed/transactions_clean.csv"

parquet_output_path = "data/processed/transactions_clean.parquet"

df.to_csv(csv_output_path, index=False)

df.to_parquet(parquet_output_path, index=False)

print("\n" + "=" * 60)
print("FILES SAVED SUCCESSFULLY")
print("=" * 60)

print(f"\nCSV File Saved To:")
print(csv_output_path)

print(f"\nParquet File Saved To:")
print(parquet_output_path)

print("\nData cleaning pipeline completed successfully.")
