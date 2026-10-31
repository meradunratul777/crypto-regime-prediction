import numpy as np
import pandas as pd

from regime_detection.cli import predict_csv
from regime_detection.models import make_model


def test_saved_bundle_predicts_csv(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4))
    y = np.tile(np.arange(4), 20)
    model = make_model("logistic", seed=7).fit(X, y)
    import joblib
    model_path = tmp_path / "model.joblib"
    joblib.dump({"schema_version": 1, "model": model, "lookback": 12,
                 "required_columns": ["open", "high", "low", "close", "volume", "trade_count", "taker_buy_volume"],
                 "feature_columns": ["bar_return", "log_range", "body", "taker_imbalance"]}, model_path)
    n = 25
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
                       "open": close, "high": close*1.01, "low": close*.99, "close": close,
                       "volume": 1., "trade_count": 10, "taker_buy_volume": .5})
    csv = tmp_path / "bars.csv"
    df.to_csv(csv, index=False)
    result = predict_csv(model_path, csv, tmp_path / "pred.csv")
    assert len(result) == 1
    assert set(result.predicted_regime).issubset({"bearish", "bullish", "quiet_range", "volatile_range"})
