# Crypto regime classification and regime-shift forecasting

`main.py` runs a full Python pipeline that predicts **range/trend regime** (not price direction) and separately forecasts whether that regime will shift within **4, 8, 12, or 16 bars**. With hourly input, these are 4/8/12/16-hour forecasts. It supports Yahoo Finance OHLCV downloads or local CSV files.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --symbol BTC-USD --interval 1h --period 730d
```

Or provide local data (recommended for reproducible, exchange-quality crypto history):

```bash
python main.py --csv path/to/ohlcv.csv --interval 1h --horizon 24 --output artifacts
```

CSV columns: `timestamp` (or `date` / `datetime`), `open`, `high`, `low`, `close`, `volume`. Timestamps are interpreted as UTC. The horizon arguments are **bars**, so 4/8/12/16 bars correspond to hours only for hourly candles. Yahoo Finance history/interval availability varies. The default `--horizon 24` sets the trailing lookback for the regime-state definition; it is not the forecast horizon.

## Targets

A causal trailing path-efficiency score is `abs(net log return) / sum(abs(one-bar log returns))` over the previous `--horizon` bars. At a threshold of 0.35 or above the state is `trend`; otherwise it is `range`. This is direction-neutral and is an operational proxy, not ground-truth market annotation.

For each forecast horizon H, the event target is positive when the causal state changes at least once between the current close and any of the next H bar closes. Four separate binary classifiers predict these events. The current regime output is also reported. No price-up/down target is trained.

## Engineering and evaluation

- Features use only current and past OHLCV: multi-lookback returns and efficiency, realized volatility and volatility-of-volatility, autocorrelation, ATR-normalized candle geometry, SMA distance in ATR units, cumulative signed volume, time since a trailing-state change, and UTC calendar terms.
- Four model families are evaluated separately for each shift horizon: Extra Trees, Random Forest, histogram gradient boosting, and regularized logistic regression. The production CLI selects via validation balanced accuracy; the expanded research benchmark uses walk-forward out-of-fold validation for model and probability-threshold selection.
- Chronological train/validation/test partitions use a 16-bar embargo, matching the maximum 16-bar future reach of the shift labels. The research benchmark purges the same gap before each fold and before the untouched final test segment.
- Imputation and scaling are fitted inside pipelines. No row shuffling, centered rolling features, or future features are used.
- Test metrics include class-conditional precision/recall/F1, average precision (PR-AUC), ROC-AUC, Brier score, log loss, calibration plots, and confusion matrices. Validation threshold tuning seeks high shift precision without ignoring recall; the held-out test is never used to tune thresholds.
- The experiment harness compares all model families against majority and no-shift persistence baselines, generates block-bootstrap intervals, sweeps lookbacks/thresholds, and audits current-regime proxy features. See [`results/expanded_benchmark.md`](results/expanded_benchmark.md) for the protocol and honest benchmark completion status.

## Outputs

The default `artifacts/` output contains `regime_models.joblib`, `metrics.json`, and `test_predictions.csv`. Run `python experiments/robustness_benchmark.py --seeds 10 --bars 900 --bootstrap 300 --outdir results/benchmark` for repeated synthetic runs, CSV tables, PR/calibration plots, permutation importance, leakage audit, and label-definition sweep. The expanded repeated-seed execution is intentionally not represented as completed until its output artifacts are generated.

## Reproducibility and limitations

`python -m pytest` runs tests for target timing and feature causality. The committed result summary is from a **synthetic smoke-test dataset, not real crypto market data**; it proves the end-to-end pipeline ran but must not be presented as expected BTC/ETH performance. Run against a representative exchange OHLCV dataset before making accuracy claims. There is no reliable way to maximize future accuracy in advance: regime labels depend on the chosen lookback/threshold and are non-stationary. For deployment, use multi-year, multi-asset exchange data, repeated walk-forward folds across market cycles, calibration and drift monitoring, and a trading simulation accounting for fees/latency. A correct shift forecast is not by itself a profitable strategy.
