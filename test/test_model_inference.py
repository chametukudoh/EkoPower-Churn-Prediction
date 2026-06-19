import numpy as np
import pandas as pd

from src.serving.inference import score_from_dataframes


class FixedProbabilityModel:
    def predict_proba(self, features):
        probabilities = np.full(len(features), 0.7)
        return np.column_stack([1 - probabilities, probabilities])


def test_dataframe_scoring_returns_probabilities_and_labels():
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
            }
        ]
    )

    result = score_from_dataframes(FixedProbabilityModel(), client, price, threshold=0.5)

    assert result.loc[0, "churn_proba"] == 0.7
    assert result.loc[0, "churn_pred"] == 1
