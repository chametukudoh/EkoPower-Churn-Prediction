# tests/test_validate_data.py
"""
Test wrapper around Great Expectations data validation.

Context:
- Uses `src/utils/validate_data.py`
- Assumes expectations suite was created already in `great_expectations/`
- Only runs smoke check, not full suite authoring
"""

import pandas as pd
import pytest

from src.utils.validate_data import validate_data
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


@pytest.fixture
def sample_dataframe():
    client_df, _ = load_data("data/raw", "train_client.csv", "train_price.csv")
    client_df = client_df.sample(n=100, random_state=1)
    return preprocess_data(client_df)


def test_validate_data_passes(sample_dataframe):
    # Run GE validation
    try:
        results = validate_data(
            df=sample_dataframe,
            suite_name="client_data_suite",
            ge_root_dir="great_expectations"
        )
        assert results["success"] is True, "Great Expectations validation failed"
    except Exception as e:
        pytest.fail(f"Validation threw unexpected error: {e}")
