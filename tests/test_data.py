import pandas as pd
import pytest

from regime_detection.data import MINUTE_COLUMNS, aggregate_complete_4h, clean_minutes


def minutes(n=480):
    ts = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    frame = pd.DataFrame({"timestamp": ts.astype("int64") // 1_000_000,
                          "open": 100., "high": 101., "low": 99., "close": 100.,
                          "volume": 1., "quote_volume": 100., "trade_count": 2,
                          "taker_buy_volume": .4})
    return frame


def test_clean_minutes_rejects_misaligned_padding_and_aggregates():
    frame = minutes()
    frame.loc[10, "timestamp"] += 1000
    cleaned, audit = clean_minutes(frame)
    assert audit["invalid_rows_removed"] == 1
    bars, detail = aggregate_complete_4h(cleaned)
    assert len(bars) == 1
    assert bars.iloc[0].minute_count == 240
    assert bars.iloc[0].trade_count == 480
    assert detail["incomplete_4h_bars_rejected"] == 1


def test_conflicting_duplicates_fail():
    frame = minutes(3)
    duplicate = frame.iloc[[1]].copy()
    duplicate.loc[:, "close"] = 100.5
    frame = pd.concat([frame, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        clean_minutes(frame)
