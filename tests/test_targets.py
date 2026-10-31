import numpy as np
import pandas as pd

from regime_detection.targets import CLASSES, momentum_baseline, regime_targets


def bars(n=150):
    index = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC", name="timestamp")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.01, n)))
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * .998,
                         "close": close, "volume": np.ones(n), "trade_count": np.ones(n),
                         "taker_buy_volume": np.full(n, .5)}, index=index)


def test_forward_label_timing_and_past_only_reference():
    frame = bars(80)
    out = regime_targets(frame, horizon=6, vol_window=42)
    t = 50
    returns = np.diff(np.log(frame.close.iloc[t - 42:t + 1].to_numpy()))
    sigma = np.sqrt(np.mean(returns**2))
    future = np.diff(np.log(frame.close.iloc[t:t + 7].to_numpy()))
    z = future.sum() / (sigma * np.sqrt(6))
    ratio = np.sqrt(np.mean(future**2)) / sigma
    expected = 0 if z <= -.5 else 1 if z >= .5 else 3 if ratio >= 1 else 2
    assert out.target.iloc[t] == expected
    assert out.decision_time.iloc[t] == frame.index[t] + pd.Timedelta(hours=4)
    assert out.target_end_time.iloc[t] == frame.index[t] + pd.Timedelta(hours=28)
    assert out.previous_regime.iloc[t] == out.target.iloc[t - 6]
    assert set(out.target.dropna().astype(int).unique()).issubset(range(len(CLASSES)))


def test_missing_grid_bar_invalidates_windows_not_filled():
    frame = bars(80).drop(bars(80).index[55])
    out = regime_targets(frame)
    assert np.isnan(out.target.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="4h", tz="UTC")).iloc[55])
    rule = momentum_baseline(frame, out)
    assert rule.isna().any()
