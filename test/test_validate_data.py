from src.utils.validate_data import _summarize_results


def test_validation_summary_reports_failed_expectations():
    ok, failures = _summarize_results(
        [
            {"success": True},
            {
                "success": False,
                "expectation_config": {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "id"},
                },
            },
        ]
    )

    assert ok is False
    assert len(failures) == 1
    assert "expect_column_values_to_not_be_null" in failures[0]
