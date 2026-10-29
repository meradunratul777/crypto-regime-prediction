#!/usr/bin/env python3
"""Leakage-aware crypto market-regime classification (range vs trend).

Run: python main.py --symbol BTC-USD --interval 1h --period 730d
Or:  python main.py --csv data.csv --timeframe 1h
CSV must contain timestamp, open, high, low, close, volume columns.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LOG = logging.getLogger("crypto_regime")
CLASSES = ["range", "trend"]


@dataclass
class Settings:
    horizon: int = 24                 # bars ahead, known only after this horizon closes
    label_threshold: float = 0.35    # path efficiency; see label_regimes
    test_fraction: float = 0.20
    validation_fraction: float = 0.20
    embargo_bars: Optional[int] = None
    random_state: int = 17


def load_data(csv_path: Optional[str], symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Load local OHLCV or fetch Yahoo Finance data (yfinance is optional)."""
    if csv_path:
        frame = pd.read_csv(csv_path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if "timestamp" not in frame:
            for candidate in ("datetime", "date", "time"):
                if candidate in frame:
                    frame = frame.rename(columns={candidate: "timestamp"})
                    break
        if "timestamp" not in frame:
            raise ValueError("CSV needs a timestamp/date column")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.set_index("timestamp")
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt`, or provide --csv") from exc
        ticker = yf.Ticker(symbol)
        frame = ticker.history(period=period, interval=interval, auto_adjust=False)
        if frame.empty:
            raise RuntimeError(f"No data returned for {symbol}; try a supported symbol/interval or --csv")
        frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
        frame.index = pd.to_datetime(frame.index, utc=True)
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    frame = frame[(frame.close > 0) & (frame.high >= frame.low)]
    if len(frame) < 500:
        raise ValueError(f"Need at least 500 valid bars; got {len(frame)}")
    return frame


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return (series - mean) / std


def make_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Causal features: every row uses current/past bars only; no centered windows."""
    o, h, l, c, v = (ohlcv[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    logc = np.log(c)
    ret = logc.diff()
    x = pd.DataFrame(index=ohlcv.index)
    # Multiple lookbacks, realized volatility, path efficiency, and return autocorrelation.
    for n in (1, 3, 6, 12, 24, 48, 72):
        x[f"return_{n}"] = logc.diff(n)
        if n >= 2:
            x[f"realized_vol_{n}"] = ret.rolling(n, min_periods=max(2, n // 2)).std() * np.sqrt(n)
        x[f"path_efficiency_{n}"] = logc.diff(n).abs() / ret.abs().rolling(n, min_periods=max(1, n // 2)).sum().replace(0, np.nan)
        if n >= 6:
            x[f"return_autocorr_{n}"] = ret.rolling(n, min_periods=max(5, n // 2)).corr(ret.shift(1))
    # True range and normalized candle geometry.
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=7).mean()
    x["atr_pct"] = atr / c
    x["range_atr"] = (h-l) / atr
    x["body_atr"] = (c-o) / atr
    x["close_location"] = ((c-l) - (h-c)) / (h-l).replace(0, np.nan)
    # Trend displacement relative to trailing price scale; indicators are causal.
    for n in (12, 24, 48, 72):
        sma = c.rolling(n, min_periods=max(3, n//2)).mean()
        x[f"sma_gap_{n}"] = c / sma - 1
        x[f"sma_slope_{n}"] = sma.pct_change(max(2, n//4))
        x[f"price_range_{n}"] = (c.rolling(n).max() - c.rolling(n).min()) / c
        x[f"sma_distance_atr_{n}"] = (c - sma) / atr
    # Added slow-horizon context: volatility-of-volatility and multi-scale signed flow.
    for n in (12, 24, 48, 72):
        vol = ret.rolling(n, min_periods=max(4, n // 2)).std()
        x[f"vol_of_vol_{n}"] = vol.rolling(n, min_periods=max(4, n // 2)).std()
        x[f"signed_volume_sum_{n}"] = (np.sign(ret) * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)
    # Volume features; robust to zero/constant volume.
    lv = np.log1p(v.clip(lower=0))
    x["volume_change"] = lv.diff()
    x["volume_z_24"] = _rolling_zscore(lv, 24)
    x["volume_z_72"] = _rolling_zscore(lv, 72)
    x["signed_volume_24"] = (np.sign(ret) * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    # UTC calendar seasonality (known at decision time).
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        hour = ohlcv.index.hour
        dow = ohlcv.index.day_of_week
        x["hour_sin"] = np.sin(2*np.pi*hour/24); x["hour_cos"] = np.cos(2*np.pi*hour/24)
        x["dow_sin"] = np.sin(2*np.pi*dow/7); x["dow_cos"] = np.cos(2*np.pi*dow/7)
    # Causal state-age proxy for long-horizon transition risk.
    x["time_since_state_change"] = time_since_state_change(ohlcv, 24, 0.35)
    return x.replace([np.inf, -np.inf], np.nan)


def label_regimes(ohlcv: pd.DataFrame, horizon: int = 24, threshold: float = 0.35) -> pd.Series:
    """Future path efficiency label, intentionally forward-looking and used only as target.

    Efficiency = absolute net log move / sum of absolute one-bar log moves over the
    next horizon. Low efficiency is range-like; high efficiency is trend-like,
    regardless of direction. Last `horizon` rows are unknown and dropped.
    """
    r = np.log(ohlcv.close.astype(float)).diff()
    future_net = np.log(ohlcv.close.astype(float)).shift(-horizon) - np.log(ohlcv.close.astype(float))
    future_path = r.abs().shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    efficiency = future_net.abs() / future_path.replace(0, np.nan)
    y = (efficiency >= threshold).astype("float")
    y[efficiency.isna()] = np.nan
    return y.rename("target")



def trailing_regime_states(ohlcv: pd.DataFrame, lookback: int = 24, threshold: float = 0.35) -> pd.Series:
    """Causal range/trend state from the trailing lookback, available at each bar close."""
    logc = np.log(ohlcv.close.astype(float))
    returns = logc.diff()
    path = returns.abs().rolling(lookback, min_periods=lookback).sum()
    efficiency = logc.diff(lookback).abs() / path.replace(0, np.nan)
    return (efficiency >= threshold).astype(float).where(efficiency.notna()).rename("state")


def time_since_state_change(ohlcv: pd.DataFrame, lookback: int = 24, threshold: float = 0.35) -> pd.Series:
    state = trailing_regime_states(ohlcv, lookback, threshold)
    changed = state.ne(state.shift(1)) & state.notna() & state.shift(1).notna()
    last = pd.Series(np.where(changed, np.arange(len(state)), np.nan), index=state.index).ffill()
    return (pd.Series(np.arange(len(state)), index=state.index) - last).rename("time_since_state_change")


def shift_targets(ohlcv: pd.DataFrame, horizons=(4, 8, 12, 16), lookback=24, threshold=0.35) -> pd.DataFrame:
    """Predict whether at least one range/trend state transition occurs within each horizon.

    State is computed causally from trailing lookback efficiency. At prediction time t,
    target H is 1 if any state from t+1 through t+H differs from the state at t.
    Labels near the end without a complete future window are NaN.
    """
    state = trailing_regime_states(ohlcv, lookback, threshold)
    targets = {}
    for h in horizons:
        # Forward rolling max over t+1..t+h; reverse rolling avoids Python row loops.
        future_states = state.shift(-1).iloc[::-1]
        changes = future_states.ne(state.iloc[::-1]) & future_states.notna() & state.iloc[::-1].notna()
        any_change = changes.astype(int).rolling(h, min_periods=h).max().iloc[::-1]
        targets[f"shift_{h}h"] = any_change
    return pd.DataFrame(targets, index=ohlcv.index)


def make_models(seed: int) -> dict:
    """Diverse nonlinear models plus regularized linear baseline, all imputation in-pipeline."""
    return {
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=8, max_features=0.75, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=2.0,
            early_stopping=False, random_state=seed)),
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(
            C=0.15, class_weight="balanced", max_iter=1500, random_state=seed)),
    }


def chronological_split(n: int, settings: Settings):
    test_start = int(n * (1-settings.test_fraction))
    val_start = int(test_start * (1-settings.validation_fraction))
    gap = settings.embargo_bars if settings.embargo_bars is not None else settings.horizon
    train_end = max(0, val_start-gap)
    val_end = max(train_end, test_start-gap)
    return slice(0, train_end), slice(val_start, val_end), slice(test_start, n)


def evaluate(y_true, pred, prob=None) -> dict:
    result = {"accuracy": float(accuracy_score(y_true, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
              "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
              "confusion_matrix": confusion_matrix(y_true, pred, labels=[0,1]).tolist(),
              "classification_report": classification_report(y_true, pred, labels=[0,1], target_names=CLASSES, zero_division=0, output_dict=True)}
    if prob is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, prob))
    return result


def run(csv: Optional[str], symbol: str, interval: str, period: str, output: str, settings: Settings) -> dict:
    data = load_data(csv, symbol, interval, period)
    X = make_features(data)
    X["time_since_state_change"] = time_since_state_change(data, settings.horizon, settings.label_threshold)
    shift_y = shift_targets(data, (4, 8, 12, 16), settings.horizon, settings.label_threshold)
    # Include current trailing regime classification alongside four future shift events.
    state = trailing_regime_states(data, settings.horizon, settings.label_threshold).rename("current_regime")
    full = pd.concat([X, shift_y, state], axis=1).dropna(subset=list(shift_y.columns) + ["current_regime"])
    X = full[X.columns]
    shift_y = full[shift_y.columns].astype(int)
    state = full.current_regime.astype(int)
    # Longest event window determines embargo so no forward target crosses partitions.
    settings.embargo_bars = max(16, settings.embargo_bars or 0)
    train_sl, val_sl, test_sl = chronological_split(len(X), settings)
    if min(train_sl.stop, val_sl.stop-val_sl.start, test_sl.stop-test_sl.start) < 50:
        raise ValueError("Not enough observations for train/validation/test after embargo; provide more data")
    candidates = make_models(settings.random_state)
    selected, selection = {}, {}
    for target in shift_y.columns:
        y = shift_y[target]
        scores = {}
        for name, model in candidates.items():
            model.fit(X.iloc[train_sl], y.iloc[train_sl])
            pred = model.predict(X.iloc[val_sl])
            # Balanced accuracy protects against the often-rare shift class.
            score = balanced_accuracy_score(y.iloc[val_sl], pred)
            scores[name] = {"validation_balanced_accuracy": float(score),
                            "validation_accuracy": float(accuracy_score(y.iloc[val_sl], pred))}
        winner = max(scores, key=lambda name: scores[name]["validation_balanced_accuracy"])
        selected[target] = winner
        selection[target] = scores
        LOG.info("%s selected %s (validation balanced accuracy %.4f)", target, winner, scores[winner]["validation_balanced_accuracy"])
    # Fit each selected specification through validation, excluding all labels whose
    # future horizon could cross into the test set. Test is evaluated once.
    import joblib
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    model_bundle, tests, prediction_data = {}, {}, pd.DataFrame(index=X.index[test_sl])
    for target in shift_y.columns:
        model = make_models(settings.random_state)[selected[target]]
        fit_end = test_sl.start - settings.embargo_bars
        y = shift_y[target]
        model.fit(X.iloc[:fit_end], y.iloc[:fit_end])
        test_y = y.iloc[test_sl]
        pred = model.predict(X.iloc[test_sl])
        prob = model.predict_proba(X.iloc[test_sl])[:, list(model.classes_).index(1)] if hasattr(model, "predict_proba") and 1 in model.classes_ else np.zeros(len(test_y))
        tests[target] = evaluate(test_y, pred, prob)
        prediction_data[f"{target}_actual"] = test_y.map({0:"no_shift", 1:"shift"})
        prediction_data[f"{target}_predicted"] = pd.Series(pred, index=test_y.index).map({0:"no_shift", 1:"shift"})
        prediction_data[f"{target}_shift_probability"] = prob
        model_bundle[target] = model
    # Current regime classification is a separate, useful output; same selected model
    # family based on mean shift-target validation score, trained on causal state labels.
    state_model_name = max(candidates, key=lambda name: np.mean([selection[t][name]["validation_balanced_accuracy"] for t in shift_y.columns]))
    state_model = make_models(settings.random_state)[state_model_name]
    state_model.fit(X.iloc[:test_sl.start-settings.embargo_bars], state.iloc[:test_sl.start-settings.embargo_bars])
    state_pred = state_model.predict(X.iloc[test_sl])
    model_bundle["current_regime"] = state_model
    tests["current_regime"] = evaluate(state.iloc[test_sl], state_pred)
    prediction_data["current_regime_actual"] = state.iloc[test_sl].map({0:"range",1:"trend"})
    prediction_data["current_regime_predicted"] = pd.Series(state_pred,index=state.iloc[test_sl].index).map({0:"range",1:"trend"})
    joblib.dump(model_bundle, out / "regime_models.joblib")
    prediction_data.to_csv(out / "test_predictions.csv", index_label="timestamp")
    report = {"data_source": "local_csv" if csv else "Yahoo Finance", "symbol": symbol, "interval": interval,
              "bars": len(data), "features": list(X.columns),
              "regime_definition": {"lookback_bars": settings.horizon, "efficiency_threshold": settings.label_threshold,
                  "efficiency": "absolute net log return / sum of absolute log returns"},
              "shift_definition": "At least one change in causal trailing range/trend state in future bars 1..H; independent of direction.",
              "split": {"train_rows": int(train_sl.stop), "validation_rows": int(val_sl.stop-val_sl.start),
                        "test_rows": int(test_sl.stop-test_sl.start), "embargo_bars": settings.embargo_bars},
              "selected_models": selected, "validation_model_comparison": selection,
              "held_out_test": tests,
              "warning": "Labels are rule-based proxies; results depend on asset, period, timeframe, lookback, and threshold. Accuracy is not profitability.",
              "test_data_note": ("Synthetic OHLCV smoke-test only; generated locally from NumPy RNG seed 2 with 1,800 hourly rows. Not real market data." if symbol == "SYNTHETIC" else "Market-data results depend on the supplied CSV or Yahoo Finance response; record source and sample dates before citing performance.")}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    LOG.info("Saved model, metrics, and test predictions to %s", out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Local OHLCV CSV; otherwise fetch Yahoo Finance")
    parser.add_argument("--symbol", default="BTC-USD", help="Yahoo ticker, e.g. BTC-USD or ETH-USD")
    parser.add_argument("--interval", default="1h", help="Yahoo interval (e.g. 1h, 1d)")
    parser.add_argument("--period", default="730d", help="Yahoo period, e.g. 730d, 5y, max")
    parser.add_argument("--timeframe", default=None, help="Alias for --interval")
    parser.add_argument("--horizon", type=int, default=24, help="Trailing lookback bars defining the range/trend state")
    parser.add_argument("--threshold", type=float, default=0.35, help="Path-efficiency threshold defining range vs trend")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.horizon < 2 or not 0 < args.threshold < 1:
        parser.error("--horizon must be >=2 and --threshold between 0 and 1")
    report = run(args.csv, args.symbol, args.timeframe or args.interval, args.period, args.output,
                 Settings(horizon=args.horizon, label_threshold=args.threshold))
    print(json.dumps({"selected_models": report["selected_models"], "held_out_test": report["held_out_test"]}, indent=2))

if __name__ == "__main__":
    main()
