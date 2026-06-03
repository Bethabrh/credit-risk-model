from src.data_processing import create_features
import pandas as pd

def test_features():
    df = pd.DataFrame({
        "CustomerId": ["C1"],
        "TransactionId": ["T1"],
        "Amount": [100],
        "Value": [100],
        "TransactionStartTime": ["2024-01-01 10:00:00"]
    })

    out = create_features(df)

assert "TransactionHour" in out.columns