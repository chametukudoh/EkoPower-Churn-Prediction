# tests/test_features.py
"""
Smoke tests for preprocessing and feature engineering in the EkoPower churn pipeline.

Context-aware:
- Mirrors expected raw columns from the base notebook:
  id, date_activ, date_end, date_modif_prod, date_renewal, has_gas, channel_sales, origin_up
  and the price table with *_var and *_fix columns for off/mid/peak.

Checks:
- preprocess_data() runs without error
- build_features() produces engineered columns (tenure_days, days_to_renewal, contract_active, contracted_tenure_years)
- One-hot encodings exist for origin_up_*/channel_sales_*
- ID and date columns are dropped post-feature-build (leakage guard)
- Price aggregations (*_mean, *_std) exist
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


def _make_client_df(n: int = 5) -> pd.DataFrame:
    today = pd.Timestamp(datetime.utcnow().date())
    rows = []
    for i in range(n):
        start = today - pd.Timedelta(days=365 + i * 10)
        end = today + pd.Timedelta(days=180 + i * 5)
        renew = today + pd.Timedelta(days=30 - i * 2)
        rows.append({
            "id": i + 1,
            "date_activ": start,
            "date_end": end,
            "date_modif_prod": start + pd.Timedelta(days=90),
            "date_renewal": renew,
            "has_gas": "t" if i % 2 == 0 else "f",
            "channel_sales": "online" if i % 2 == 0 else "retail",
            "origin_up": "us_market" if i % 3 == 0 else "eu_market",
            # target included to ensure it is not required by build_features
            "has_churned": 1 if i % 4 == 0 else 0,
        })
    return pd.DataFrame(rows)


def _make_price_df(n_ids: int = 5, hist_per_id: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for i in range(1, n_ids + 1):
        for _ in range(hist_per_id):
            rows.append({
                "id": i,
                "price_off_peak_var": float(rng.uniform(0.05, 0.15)),
                "price_peak_var": float(rng.uniform(0.15, 0.35)),
                "price_mid_peak_var": float(rng.uniform(0.10, 0.25)),
                "price_off_peak_fix": float(rng.uniform(0.01, 0.05)),
                "price_peak_fix": float(rng.uniform(0.02, 0.08)),
                "price_mid_peak_fix": float(rng.uniform(0.015, 0.06)),
            })
    return pd.DataFrame(rows)


def test_preprocess_and_build_features_end_to_end():
    # Arrange
    client_df = _make_client_df(7)
    price_df = _make_price_df(7, 4)

    # Act
    client_clean = preprocess_data(client_df.copy())
    feat_df = build_features(client_clean, price_df.copy())

    # Assert — engineered columns exist
    assert "tenure_days" in feat_df.columns, "Expected tenure_days to be present"
    assert "days_to_renewal" in feat_df.columns, "Expected days_to_renewal to be present"
    assert "contract_active" in feat_df.columns, "Expected contract_active to be present"
    assert "contracted_tenure_years" in feat_df.columns, "Expected contracted_tenure_years to be present"

    # Assert — binary mapping preserved for has_gas after processing
    if "has_gas" in feat_df.columns:
        unique_vals = set(pd.Series(feat_df["has_gas"]).dropna().unique().tolist())
        assert unique_vals.issubset({0, 1}), f"has_gas should be binary after processing, got: {unique_vals}"

    # Assert — one-hot encodings
    assert any(c.startswith("origin_up_") for c in feat_df.columns), "Expected origin_up_* one-hot columns"
    assert any(c.startswith("channel_sales_") for c in feat_df.columns), "Expected channel_sales_* one-hot columns"

    # Assert — leakage guard: id and raw date columns should be dropped
    forbidden = {"id", "date_activ", "date_end", "date_modif_prod", "date_renewal"}
    assert forbidden.isdisjoint(set(feat_df.columns)), f"Forbidden leakage columns present: {forbidden & set(feat_df.columns)}"

    # Assert — price aggregations exist
    expected_prefixes = [
        "price_off_peak_var_", "price_peak_var_", "price_mid_peak_var_",
        "price_off_peak_fix_", "price_peak_fix_", "price_mid_peak_fix_",
    ]
    price_agg_cols = [c for c in feat_df.columns if any(c.startswith(p) for p in expected_prefixes)]
    assert len(price_agg_cols) > 0, "Expected price aggregation columns (*_mean, *_std) to be present"

    # Sanity — no empty dataframe
    assert len(feat_df) == len(client_df), "Feature DF should preserve row count per client id merge (left join on id)"
