# tests/test_data_load.py
"""
Smoke test for data loading from local storage.

Covers:
- client data (client_data.csv)
- pricing data (price_data.csv)

Context-aware:
- Expects these files to exist in `data/raw/`
- Validates basic structure: non-empty, key columns, and ID consistency
"""

import os
import pandas as pd

from src.data.load_data import load_data

RAW_PATH = "data/raw"
CLIENT_FILE = "client_data.csv"
PRICE_FILE = "price_data.csv"


def test_data_files_exist():
    assert os.path.exists(os.path.join(RAW_PATH, CLIENT_FILE)), f"{CLIENT_FILE} not found in {RAW_PATH}"
    assert os.path.exists(os.path.join(RAW_PATH, PRICE_FILE)), f"{PRICE_FILE} not found in {RAW_PATH}"


def test_load_data_shapes():
    client_df, price_df = load_data(RAW_PATH, CLIENT_FILE, PRICE_FILE)

    assert not client_df.empty, "Client data is empty"
    assert not price_df.empty, "Price data is empty"

    # Check reasonable shape
    assert len(client_df.columns) > 5, "Client data has too few columns"
    assert len(price_df.columns) > 5, "Price data has too few columns"


def test_required_columns_exist():
    client_df, price_df = load_data(RAW_PATH, CLIENT_FILE, PRICE_FILE)

    expected_client_cols = {"id", "date_activ", "date_end", "has_gas", "origin_up", "has_churned"}
    expected_price_cols = {"id", "price_off_peak_var", "price_peak_fix", "price_mid_peak_fix"}

    assert expected_client_cols.issubset(set(client_df.columns)), f"Missing client columns: {expected_client_cols - set(client_df.columns)}"
    assert expected_price_cols.issubset(set(price_df.columns)), f"Missing price columns: {expected_price_cols - set(price_df.columns)}"


def test_id_overlap_for_merge():
    client_df, price_df = load_data(RAW_PATH, CLIENT_FILE, PRICE_FILE)

    client_ids = set(client_df["id"].unique())
    price_ids = set(price_df["id"].unique())
    overlap = client_ids & price_ids

    assert len(overlap) > 0, "No overlapping IDs between client and price data"
