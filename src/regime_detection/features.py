"""Causal, scale-free OHLCV features. Every rolling statistic is right-aligned.

A lookback L restricts all price/volume features to the current L completed 4H
candles. Feature families are nested as L grows. No target, future return,
forward volatility, global z-score, centered window, or fitted test statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .targets import regular_grid

EPS = 1e-12


def _roll_slope(values: pd.Series, window: int) -> pd.Series:
    """Vectorized OLS slope against time, normalized by price and window."""
    weights = np.arange(window, dtype=float)
    weights -= weights.mean()
    denominator = float(np.dot(weights, weights))
    slope = values.rolling(window, min_periods=window).apply(lambda x: np.dot(weights, x) / denominator, raw=True)
    return slope / values.rolling(window, min_periods=window).mean().clip(lower=EPS)


def make_features(bars: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 12:
        raise ValueError("Minimum lookback is 12 completed 4H candles")
    grid = regular_grid(bars)
    o, h, l, c, v = (grid[name] for name in ["open", "high", "low", "close", "volume"])
    logc = np.log(c)
    r = logc.diff()
    feature_columns = {}

    def add(name: str, value) -> None:
        feature_columns[name] = value

    log_range = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    body = (c - o) / o
    upper_wick = (h - np.maximum(o, c)) / o
    lower_wick = (np.minimum(o, c) - l) / o
    close_location = (c - l) / (h - l).replace(0, np.nan)
    true_range = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1) / c
    signed_volume = v * np.sign(r)
    taker_imbalance = 2 * grid["taker_buy_volume"] / v.replace(0, np.nan) - 1
    log_volume = np.log1p(v)
    log_trades = np.log1p(grid["trade_count"])
    # The primitive window is one completed candle, not a future statistic.
    for name, value in {"bar_return": r, "overnight_gap": np.log(o / c.shift()),
                        "log_range": log_range, "body": body, "upper_wick": upper_wick,
                        "lower_wick": lower_wick, "close_location": close_location - 0.5,
                        "true_range": true_range, "taker_imbalance": taker_imbalance}.items():
        add(name, value)

    # Include fixed short scales and fractions of the experimental lookback.
    # w return intervals need w+1 candles, so w <= L-1 enforces the L-candle cap.
    windows = sorted({w for w in (3, 6, 12, 18, 30, 42, 60, 90, 120, 180,
                                  lookback // 4, lookback // 2, lookback - 1) if 2 <= w <= lookback - 1})
    for w in windows:
        rr = r.rolling(w, min_periods=w)
        mean = rr.mean()
        std = rr.std(ddof=0).clip(lower=1e-8)
        net = logc.diff(w)
        add(f"return_{w}", net)
        add(f"volatility_{w}", std)
        add(f"downside_volatility_{w}", np.sqrt(r.clip(upper=0).pow(2).rolling(w, min_periods=w).mean()))
        add(f"efficiency_{w}", net.abs() / r.abs().rolling(w, min_periods=w).sum().clip(lower=EPS))
        add(f"positive_fraction_{w}", (r > 0).rolling(w, min_periods=w).mean())
        add(f"return_skew_{w}", r.rolling(w, min_periods=w).skew())
        add(f"return_kurtosis_{w}", r.rolling(w, min_periods=w).kurt())
        corr_window = max(3, w - 1)  # shifted pair adds one return to the history
        add(f"return_autocorr_{w}", r.rolling(corr_window, min_periods=corr_window).corr(r.shift(1)))
        add(f"trend_slope_{w}", _roll_slope(c, w))
        add(f"range_volatility_{w}", np.sqrt(log_range.pow(2).rolling(w, min_periods=w).mean() / (4 * np.log(2))))
        add(f"atr_{w}", true_range.rolling(w, min_periods=w).mean())
        rolling_mean = c.rolling(w, min_periods=w).mean()
        rolling_std = c.rolling(w, min_periods=w).std(ddof=0).clip(lower=EPS)
        add(f"price_zscore_{w}", (c - rolling_mean) / rolling_std)
        add(f"sma_distance_{w}", (c / rolling_mean - 1) / std)
        hi, lo = h.rolling(w, min_periods=w).max(), l.rolling(w, min_periods=w).min()
        add(f"range_position_{w}", (c - lo) / (hi - lo).clip(lower=EPS) - 0.5)
        add(f"drawdown_{w}", c / hi - 1)
        add(f"runup_{w}", c / lo - 1)
        add(f"max_return_{w}", r.rolling(w, min_periods=w).max())
        add(f"min_return_{w}", r.rolling(w, min_periods=w).min())
        add(f"volume_zscore_{w}", (log_volume - log_volume.rolling(w, min_periods=w).mean()) /
            log_volume.rolling(w, min_periods=w).std(ddof=0).clip(lower=EPS))
        add(f"trade_count_zscore_{w}", (log_trades - log_trades.rolling(w, min_periods=w).mean()) /
            log_trades.rolling(w, min_periods=w).std(ddof=0).clip(lower=EPS))
        add(f"signed_volume_{w}", signed_volume.rolling(w, min_periods=w).sum() / v.rolling(w, min_periods=w).sum().clip(lower=EPS))
        add(f"taker_imbalance_{w}", taker_imbalance.rolling(w, min_periods=w).mean())
        add(f"price_volume_correlation_{w}", r.rolling(corr_window, min_periods=corr_window).corr(log_volume.diff()))
        add(f"body_mean_{w}", body.rolling(w, min_periods=w).mean())
        add(f"close_location_mean_{w}", close_location.rolling(w, min_periods=w).mean())
        # Nested rolling windows must also fit inside the L-candle budget.
        inner = max(2, w // 3)
        outer = w - inner + 1
        vol = np.sqrt(r.pow(2).rolling(inner, min_periods=inner).mean())
        add(f"volatility_of_volatility_{w}", vol.rolling(outer, min_periods=outer).std(ddof=0) / vol.rolling(outer, min_periods=outer).mean().clip(lower=EPS))
        if w >= 6:
            add(f"volatility_ratio_6_{w}", np.sqrt(r.pow(2).rolling(6, min_periods=6).mean()) /
                np.sqrt(r.pow(2).rolling(w, min_periods=w).mean()).clip(lower=EPS))
            add(f"volume_ratio_6_{w}", np.log1p(v.rolling(6, min_periods=6).mean()) /
                np.log1p(v.rolling(w, min_periods=w).mean()).clip(lower=EPS))
    # Recent path shape, with the normalization history included in the L-bar cap.
    for lag in range(1, min(12, lookback // 2) + 1):
        sigma_window = min(42, lookback - lag)
        base_sigma = np.sqrt(r.pow(2).rolling(sigma_window, min_periods=sigma_window).mean()).clip(lower=1e-8)
        add(f"normalized_return_lag_{lag}", r.shift(lag - 1) / base_sigma)
    # Time-of-day/week use the candle close time, never the target's future date.
    close_time = grid.index + pd.Timedelta(hours=4)
    hour = close_time.hour + close_time.minute / 60
    dow = close_time.dayofweek.to_numpy()
    add("hour_sin", np.sin(2 * np.pi * hour / 24))
    add("hour_cos", np.cos(2 * np.pi * hour / 24))
    add("week_sin", np.sin(2 * np.pi * dow / 7))
    add("week_cos", np.cos(2 * np.pi * dow / 7))
    features = pd.DataFrame(feature_columns, index=grid.index).replace([np.inf, -np.inf], np.nan)
    # Drop incomplete histories, including missing-grid candles, instead of
    # silently treating a long outage as a shorter lookback.
    complete = grid["close"].notna().rolling(lookback, min_periods=lookback).sum().eq(lookback)
    features.loc[~complete] = np.nan
    features.index.name = "timestamp"
    return features


def feature_group_ablation(columns: list[str], group: str) -> list[str]:
    """Predeclared validation-only ablations; test never chooses the feature set."""
    volume_terms = ("volume", "trade_count", "taker")
    if group == "price_only":
        return [c for c in columns if not any(term in c for term in volume_terms)]
    if group == "no_calendar":
        return [c for c in columns if not c.startswith(("hour_", "week_"))]
    return list(columns)
