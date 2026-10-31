"""Explicit, fixed forward-window regime definition (no learned/test thresholds)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BAR

CLASSES = ["bearish", "bullish", "quiet_range", "volatile_range"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def regular_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Preserve gaps as NaNs: never turn 12 elapsed hours into one 4H return."""
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Candles must have unique, sorted timestamps")
    if bars.index.tz is None or str(bars.index.tz) != "UTC":
        raise ValueError("Candles must use UTC")
    if (bars.index.asi8 % int(BAR.value) != 0).any():
        raise ValueError("Candles must align to UTC 4-hour boundaries")
    return bars.reindex(pd.date_range(bars.index.min(), bars.index.max(), freq=BAR, name="timestamp"))


def regime_targets(bars: pd.DataFrame, horizon: int = 6, vol_window: int = 42,
                   direction_threshold: float = 0.5, range_vol_threshold: float = 1.0) -> pd.DataFrame:
    """Label the NEXT horizon bars, observable only horizon*4 hours after decision.

    sigma is RMS of the previous vol_window one-bar log returns. A forward net
    move >= 0.5*sigma*sqrt(H) is bullish/bearish. Otherwise, forward RMS >= sigma
    is volatile range; below it is quiet range. Thresholds are fixed before test.
    Targets are an operational definition, not objective market ground truth.
    """
    if horizon < 1 or vol_window < 2 or direction_threshold <= 0 or range_vol_threshold <= 0:
        raise ValueError("Invalid target parameters")
    grid = regular_grid(bars)
    ret = np.log(grid["close"]).diff()
    ret.loc[grid["close"].isna()] = np.nan
    sigma = np.sqrt(ret.pow(2).rolling(vol_window, min_periods=vol_window).mean()).clip(lower=1e-8)
    future_return = np.log(grid["close"].shift(-horizon) / grid["close"])
    future_rms = np.sqrt(ret.shift(-1).pow(2).rolling(horizon, min_periods=horizon).mean().shift(-(horizon - 1)))
    # A label requires every future candle and every preceding sigma return.
    future_complete = grid["close"].shift(-1).notna().rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1)).eq(horizon)
    past_complete = grid["close"].notna().rolling(vol_window + 1, min_periods=vol_window + 1).sum().eq(vol_window + 1)
    valid = future_complete & past_complete & sigma.notna() & future_rms.notna()
    z = future_return / (sigma * np.sqrt(horizon))
    vol_ratio = future_rms / sigma
    labels = pd.Series(np.select([z <= -direction_threshold, z >= direction_threshold,
                                  vol_ratio >= range_vol_threshold], [0, 1, 3], default=2), index=grid.index, dtype=float)
    labels.loc[~valid] = np.nan
    result = pd.DataFrame({"target": labels, "sigma_reference": sigma,
                           "future_normalized_return": z, "future_volatility_ratio": vol_ratio})
    result["decision_time"] = result.index + BAR
    result["target_end_time"] = result["decision_time"] + horizon * BAR
    result["previous_regime"] = labels.shift(horizon)
    return result


def momentum_baseline(bars: pd.DataFrame, targets: pd.DataFrame, horizon: int = 6,
                      direction_threshold: float = 0.5, range_vol_threshold: float = 1.0) -> pd.Series:
    """Apply the same economic rule to the immediately preceding, completed window."""
    grid = regular_grid(bars)
    ret = np.log(grid["close"]).diff()
    past_return = np.log(grid["close"] / grid["close"].shift(horizon))
    past_rms = np.sqrt(ret.pow(2).rolling(horizon, min_periods=horizon).mean())
    sigma = targets["sigma_reference"]
    z = past_return / (sigma * np.sqrt(horizon))
    prediction = pd.Series(np.select([z <= -direction_threshold, z >= direction_threshold,
                                     past_rms / sigma >= range_vol_threshold], [0, 1, 3], default=2), index=grid.index, dtype=float)
    prediction.loc[z.isna() | past_rms.isna() | sigma.isna()] = np.nan
    return prediction
