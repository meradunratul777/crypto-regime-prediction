import numpy as np
import pandas as pd

from main import make_features, shift_targets, trailing_regime_states


def _ohlcv(close):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999, "close": close,
                         "volume": np.ones(len(close))}, index=idx)


def test_shift_targets_match_definition_and_truncate_future():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    data = _ohlcv(close)
    state = trailing_regime_states(data, lookback=12, threshold=0.4)
    labels = shift_targets(data, horizons=(4, 8), lookback=12, threshold=0.4)
    for h in (4, 8):
        col = f"shift_{h}h"
        for i in range(20, 480):
            expected = int(any(state.iloc[i + j] != state.iloc[i] for j in range(1, h + 1)))
            assert labels[col].iloc[i] == expected
        assert labels[col].iloc[-(h - 1):].isna().all()


def test_features_are_causal():
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
    first, changed = _ohlcv(close), _ohlcv(close.copy())
    changed.iloc[250:, changed.columns.get_loc("close")] *= 2
    # Mutate future OHLC fields consistently enough to prove earlier features don't move.
    changed.iloc[250:, changed.columns.get_loc("high")] *= 2
    changed.iloc[250:, changed.columns.get_loc("low")] *= 2
    changed.iloc[250:, changed.columns.get_loc("open")] *= 2
    a, b = make_features(first), make_features(changed)
    pd.testing.assert_frame_equal(a.iloc[:250], b.iloc[:250])
