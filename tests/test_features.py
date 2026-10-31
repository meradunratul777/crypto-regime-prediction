import numpy as np
import pandas as pd

from regime_detection.features import make_features
from test_targets import bars


def test_features_are_prefix_causal_and_lookback_validated():
    frame = bars(260)
    baseline = make_features(frame, 60)
    changed = frame.copy()
    changed.loc[changed.index[180]:, ["open", "high", "low", "close", "volume", "trade_count", "taker_buy_volume"]] *= 3
    perturbed = make_features(changed, 60)
    pd.testing.assert_frame_equal(baseline.loc[:frame.index[179]], perturbed.loc[:frame.index[179]])
    assert baseline.shape[1] > 100


def test_feature_builder_rejects_too_short_lookback():
    import pytest
    with pytest.raises(ValueError):
        make_features(bars(30), 11)


def test_gap_invalidates_history_and_recovery_is_bounded():
    frame = bars(100)
    frame = frame.drop(frame.index[60])
    features = make_features(frame, 12)
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="4h", tz="UTC")
    aligned = features.reindex(full_index)
    assert aligned.iloc[60:72].isna().all(axis=None)
    assert aligned.iloc[72:].notna().any(axis=1).any()
