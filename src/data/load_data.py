"""
Load raw datasets for the EkoPower churn prediction pipeline.

This script ensures:
1. Proper file path resolution
2. Data integrity checks
3. Consistent return of pandas DataFrames
"""

import os
import pandas as pd


def load_data(client_path: str = "data/client_data.csv",
              price_path: str = "data/price_data.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw client and price datasets.

    Args:
        client_path (str): Path to the client dataset.
        price_path (str): Path to the price dataset.

    Returns:
        tuple: (client_df, price_df)
    """

    # === 1️⃣ Verify paths exist ===
    if not os.path.exists(client_path):
        raise FileNotFoundError(f"❌ Client dataset not found at: {client_path}")

    if not os.path.exists(price_path):
        raise FileNotFoundError(f"❌ Price dataset not found at: {price_path}")

    # === 2️⃣ Load CSVs ===
    print(f"Loading client data from: {client_path}")
    client_df = pd.read_csv(client_path)

    print(f"Loading price data from: {price_path}")
    price_df = pd.read_csv(price_path)

    # === 3️⃣ Quick validation ===
    if client_df.empty:
        raise ValueError("Client dataset is empty.")
    if price_df.empty:
        raise ValueError("Price dataset is empty.")

    print(f" Client data shape: {client_df.shape}")
    print(f" Price data shape: {price_df.shape}")

    return client_df, price_df

if __name__ == "__main__":
    client_df, price_df = load_data()
    print("Client data preview:")
    print(client_df.head())
    print("\nPrice data preview:")
    print(price_df.head())