import pandas as pd


def preprocess_data(client_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess client and price data for churn prediction.
    - Clean headers
    - Parse date fields
    - Derive tenure, renewal, and contract status
    - Aggregate price vars by ID
    - Encode binary fields
    - Dummify selected categoricals
    - Drop unneeded or leakage-prone columns
    """

    # 1. Tidy headers
    client_df.columns = client_df.columns.str.strip()
    price_df.columns = price_df.columns.str.strip()

    # 2. Parse key date fields
    date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']
    for col in date_columns:
        if col in client_df.columns:
            client_df[col] = pd.to_datetime(client_df[col], errors="coerce")

    # 3. Create date-based features
    today = pd.Timestamp('today').normalize()
    client_df["tenure_days"] = (today - client_df["date_activ"]).dt.days
    client_df["days_to_renewal"] = (client_df["date_renewal"] - today).dt.days
    client_df["contract_active"] = (client_df["date_end"] > today).astype(int)
    client_df["contracted_tenure_years"] = (
        (client_df["date_end"] - client_df["date_activ"]).dt.days / 365.25
    ).round().astype("Int64")

    # 4. Price aggregations by customer ID
    price_vars = [
        "price_off_peak_var", "price_peak_var", "price_mid_peak_var",
        "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix"
    ]
    price_agg = price_df.groupby("id")[price_vars].agg(["mean", "std"])
    price_agg.columns = ["_".join(col) for col in price_agg.columns]
    price_agg = price_agg.reset_index()

    # 5. Merge price stats into client data
    df = pd.merge(client_df, price_agg, on="id", how="left")

    # 6. Binary flag: has_gas (from 't'/'f' to 1/0)
    if "has_gas" in df.columns:
        df["has_gas"] = df["has_gas"].map({"t": 1, "f": 0})

    # 7. Encode categorical columns
    for col in ["channel_sales", "origin_up"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 8. Dummify origin_up and Channel_sales (and drop leakage/missing cols if needed)
    if "origin_up" in df.columns:
        df = pd.get_dummies(df, columns=["origin_up"], prefix="origin_up", dummy_na=True)
        drop_cols = [c for c in df.columns if "origin_up_MISSING" in c or "usapbepcfoloekilkwsdib" in c]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
    
    if "channel_sales" in df.columns:
        df = pd.get_dummies(df, columns=["channel_sales"], prefix="channel_sales", dummy_na=True)
        drop_colss = [c for c in df.columns if 'channel_sddiedcslfslkckwlfkdpoeeailfpeds' in c or 'channel_epumfxlbckeskwekxbiuasklxalciiuu'in c or 'channel_fixdbufsefwooaasfcxdxadsiekoceaa' in c]
        df.drop(columns=drop_colss, inplace=True, errors="ignore")


    # 9. Optional: drop ID and raw date columns to prevent leakage
    df.drop(columns=["id"] + date_columns, inplace=True, errors="ignore")

    return df
