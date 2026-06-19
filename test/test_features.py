import pandas as pd

from src.data.preprocess import preprocess_data


def test_preprocess_builds_dates_prices_and_categories():
    client = pd.DataFrame(
        [
            {
                "id": "a",
                "date_activ": "2024-01-01",
                "date_end": "2027-01-01",
                "date_modif_prod": "2025-01-01",
                "date_renewal": "2026-12-01",
                "has_gas": "t",
                "channel_sales": "online",
                "origin_up": "campaign",
                "churn": 0,
            }
        ]
    )
    price = pd.DataFrame(
        [
            {
                "id": "a",
                "price_off_peak_var": 0.1,
                "price_peak_var": 0.2,
                "price_mid_peak_var": 0.15,
                "price_off_peak_fix": 1.0,
                "price_peak_fix": 2.0,
                "price_mid_peak_fix": 1.5,
            },
            {
                "id": "a",
                "price_off_peak_var": 0.12,
                "price_peak_var": 0.22,
                "price_mid_peak_var": 0.17,
                "price_off_peak_fix": 1.2,
                "price_peak_fix": 2.2,
                "price_mid_peak_fix": 1.7,
            },
        ]
    )

    result = preprocess_data(client, price)

    assert "tenure_days" in result
    assert "price_off_peak_var_mean" in result
    assert any(column.startswith("channel_sales_") for column in result)
    assert result.loc[0, "has_gas"] == 1
    assert "id" not in result
    assert "date_activ" not in result
