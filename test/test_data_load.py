from src.data.load_data import load_data


def test_repository_data_loads_and_has_merge_keys():
    client, price = load_data("data/client_data.csv", "data/price_data.csv")

    assert not client.empty
    assert not price.empty
    assert {"id", "date_activ", "date_end", "churn"}.issubset(client.columns)
    assert {"id", "price_off_peak_var", "price_peak_fix"}.issubset(price.columns)
    assert set(client["id"]).intersection(price["id"])
