# src/utils/validate_data.py

"""
Great Expectations-based validators for the EkoPower churn pipeline.

- validate_client_df: raw client table sanity checks
- validate_price_df:  raw price table sanity checks
- validate_feature_df: post-merge / engineered feature checks

Uses the lightweight ge.from_pandas() API so you can call these
directly in notebooks/scripts without initializing a GE project.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import great_expectations as ge


# ---------- helpers ----------

def _summarize_results(results: List[Dict]) -> Tuple[bool, List[str]]:
    """Aggregate GE expectation results into a single success flag and error messages."""
    failed = []
    for r in results:
        ok = r.get("success", False)
        if not ok:
            cfg = r.get("expectation_config", {})
            exp_type = cfg.get("expectation_type", "unknown_expectation")
            kwargs = cfg.get("kwargs", {})
            failed.append(f"{exp_type} failed with kwargs={kwargs}")
    return (len(failed) == 0), failed


def _raise_if_failed(results: List[Dict], context: str) -> None:
    ok, failed = _summarize_results(results)
    if not ok:
        msg = f"[{context}] Data validation failed:\n- " + "\n- ".join(failed)
        raise ValueError(msg)


# ---------- validators ----------

def validate_client_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the *raw* client dataframe BEFORE preprocessing.

    Expected columns (based on notebook):
      - id (unique key per client)
      - date_activ, date_end, date_modif_prod, date_renewal (dates or parseable)
      - has_gas (expected 't'/'f' in raw), channel_sales, origin_up
    """
    gdf = ge.from_pandas(df.copy())

    expected_cols = [
        "id",
        "date_activ", "date_end", "date_modif_prod", "date_renewal",
        "has_gas", "channel_sales", "origin_up"
    ]

    results = []
    # presence
    results.append(gdf.expect_table_columns_to_match_set(column_set=expected_cols, exact_match=False))
    # id checks
    if "id" in df.columns:
        results.append(gdf.expect_column_values_to_not_be_null("id"))
        results.append(gdf.expect_column_values_to_be_unique("id"))
    # gas flag (raw)
    if "has_gas" in df.columns and df["has_gas"].dtype == object:
        results.append(gdf.expect_column_values_to_be_in_set("has_gas", ["t", "f", "T", "F"]))
    # basic null thresholds on critical cols
    for c in ["date_activ", "date_end", "date_modif_prod", "date_renewal", "channel_sales", "origin_up"]:
        if c in df.columns:
            results.append(gdf.expect_column_values_to_not_be_null(c, mostly=0.95))

    _raise_if_failed(results, "client_df")
    return df


def validate_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the *raw* price dataframe BEFORE preprocessing.

    Expected columns (based on notebook):
      - id (foreign key)
      - price_off_peak_var, price_peak_var, price_mid_peak_var
      - price_off_peak_fix, price_peak_fix, price_mid_peak_fix
    """
    gdf = ge.from_pandas(df.copy())

    price_cols = [
        "price_off_peak_var", "price_peak_var", "price_mid_peak_var",
        "price_off_peak_fix", "price_peak_fix", "price_mid_peak_fix"
    ]
    expected_cols = ["id"] + price_cols

    results = []
    results.append(gdf.expect_table_columns_to_match_set(column_set=expected_cols, exact_match=False))

    # id presence (can repeat because it’s a history table), but should not be null
    if "id" in df.columns:
        results.append(gdf.expect_column_values_to_not_be_null("id"))

    # numeric & non-negative checks
    for c in price_cols:
        if c in df.columns:
            results.append(gdf.expect_column_values_to_not_be_null(c, mostly=0.99))
            results.append(gdf.expect_column_values_to_be_of_type(c, "float64", mostly=0.8))
            results.append(gdf.expect_column_min_to_be_between(c, min_value=0, allow_cross_type_comparisons=True))

    _raise_if_failed(results, "price_df")
    return df


def validate_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the FEATURE dataframe AFTER preprocess + build_features merge.

    Expected engineered columns (from notebook):
      - tenure_days (>= 0)
      - days_to_renewal (can be negative if past due; allow a lower bound e.g., >= -36500)
      - contract_active in {0,1}
      - contracted_tenure_years (nullable Int; non-negative)
      - has_gas mapped to {0,1} (post-processing)
      - aggregated price features: *_mean, *_std
      - one-hot columns for origin_up_* and channel_sales_*
    """
    gdf = ge.from_pandas(df.copy())

    results = []

    # engineered numeric columns
    if "tenure_days" in df.columns:
        results.append(gdf.expect_column_values_to_not_be_null("tenure_days", mostly=0.99))
        results.append(gdf.expect_column_min_to_be_between("tenure_days", min_value=0, allow_cross_type_comparisons=True))

    if "days_to_renewal" in df.columns:
        results.append(gdf.expect_column_values_to_not_be_null("days_to_renewal", mostly=0.98))
        # allow large negative (up to 100 years) in case of data quirks, and cap upper bound generously
        results.append(gdf.expect_column_values_to_be_between("days_to_renewal", min_value=-36500, max_value=36500, allow_cross_type_comparisons=True))

    if "contract_active" in df.columns:
        results.append(gdf.expect_column_values_to_be_in_set("contract_active", [0, 1, np.int64(0), np.int64(1)]))

    if "contracted_tenure_years" in df.columns:
        # Can be nullable Int; check non-negatives where present
        non_null = df["contracted_tenure_years"].dropna()
        if not non_null.empty:
            results.append(gdf.expect_column_min_to_be_between("contracted_tenure_years", min_value=0, allow_cross_type_comparisons=True))

    if "has_gas" in df.columns and df["has_gas"].dtype != object:
        results.append(gdf.expect_column_values_to_be_in_set("has_gas", [0, 1, np.int64(0), np.int64(1)]))

    # aggregated price columns from build_features
    price_agg_prefixes = [
        "price_off_peak_var_", "price_peak_var_", "price_mid_peak_var_",
        "price_off_peak_fix_", "price_peak_fix_", "price_mid_peak_fix_",
    ]
    price_agg_cols = [c for c in df.columns if any(c.startswith(p) for p in price_agg_prefixes)]
    for c in price_agg_cols:
        results.append(gdf.expect_column_values_to_be_of_type(c, "float64", mostly=0.7))

    # one-hot encodings
    ohe_origin_cols = [c for c in df.columns if c.startswith("origin_up_")]
    ohe_channel_cols = [c for c in df.columns if c.startswith("channel_sales_")]
    for c in ohe_origin_cols + ohe_channel_cols:
        results.append(gdf.expect_column_values_to_be_in_set(c, [0, 1, np.int64(0), np.int64(1)], mostly=1.0))

    # leakage guard: dates & id should have been dropped by build_features
    forbidden_cols = {"id", "date_activ", "date_end", "date_modif_prod", "date_renewal"}
    present_forbidden = forbidden_cols.intersection(set(df.columns))
    results.append({
        "success": len(present_forbidden) == 0,
        "expectation_config": {
            "expectation_type": "expect_forbidden_columns_absent",
            "kwargs": {"forbidden_columns_present": sorted(list(present_forbidden))}
        }
    })

    _raise_if_failed(results, "feature_df")
    return df
