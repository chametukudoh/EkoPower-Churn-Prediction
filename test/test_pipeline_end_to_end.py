from src.data.load_data import load_data
from src.data.preprocess import preprocess_data


def test_repository_data_preprocesses_end_to_end():
    client, price = load_data("data/client_data.csv", "data/price_data.csv")
    client = client.head(200).copy()
    price = price[price["id"].isin(client["id"])].copy()

    result = preprocess_data(client, price)

    assert len(result) == len(client)
    assert "churn" in result
    assert "id" not in result
    assert "date_renewal" not in result
    assert any(column.endswith("_mean") for column in result)
