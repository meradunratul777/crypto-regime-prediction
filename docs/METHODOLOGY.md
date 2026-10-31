# Methodology and audit notes

## Data engineering

Inputs are pinned monthly one-minute BTCUSDT and ETHUSDT spot-kline archives from third-party GitHub mirrors. These are not exchange-certified, and the source repositories do not report a data license. This project therefore does not redistribute raw archives or processed market-price files; users should review source terms before use. `configs/sources.json` records repositories and full commit IDs. The archive files and every selected source CSV are SHA256-hashed; `data/manifest.json` stores source identity, per-file audit counts/hashes, and output parquet hashes.

Minute timestamps are converted from milliseconds/microseconds, but never rounded to repair misalignment. Rows with sub-minute timestamps, invalid OHLC relationships, non-finite/negative quantities, or invalid taker-buy volume are excluded. Identical duplicate rows can be dropped; conflicting duplicate timestamps fail loudly. Four-hour bars are emitted only with 240 distinct valid minutes. Open=first, high=max, low=min, close=last, volumes/trades=sum. No prices, volume, or missing bars are imputed. Gaps stay gaps in all rolling calculations.

The committed dataset covers 2017-12-01 through 2026-01-01 inclusive: 17,571 complete 4H bars for each asset. It rejected 21,602 invalid/padding minute rows and 153 incomplete 4H bars per symbol. The source is a mirror, and its license/provenance limitations are documented in the data manifest.

## Target and observation timing

Bars are timestamped by UTC candle **open**. A bar at timestamp `t` is known at `t+4h`; the classifier decision occurs at that close. Features are computed from data through that completed candle. The next six returns define the 24-hour target, which resolves at `decision_time+24h`. Training labels whose resolution crosses a validation/test boundary are purged. The prior-regime persistence feature is shifted six candles so its full label window has resolved by the current decision time.

The fixed target uses RMS of the preceding 42 log returns as volatility scale. The future net return normalized by `sigma*sqrt(6)` determines bearish/bullish at thresholds +/-0.5. Non-directional windows with future return RMS at least `sigma` are `volatile_range`; the rest are `quiet_range`. This definition is fixed before testing, handles gaps as missing labels, and is not an external market-state annotation.

## Features and lookbacks

Features use completed OHLCV, trade-count, and taker-buy information: log returns, range/body/wicks, true range, directional efficiency, realized/downside/range volatility, trend slopes, drawdown/run-up, price range location, return distribution moments, autocorrelation, volume/taker imbalance, volume/trade z-scores, normalized lag returns, and UTC cyclical time features. All windows are right aligned and scale free. No centered windows, raw price level, global normalization, target values, or future features.

Lookbacks are 12, 30, 60, 120, and 180 completed candles (2, 5, 10, 20, and 30 days). Nested and shifted-pair computations are explicitly budgeted within the chosen lookback. To ensure a fair comparison, every lookback is evaluated on the same eligible decision rows, requiring the longest lookback and all baseline labels to be available.

## Model search and metrics

Four predefined estimators are compared: LightGBM, histogram gradient boosting, ExtraTrees, and regularized logistic regression. For each lookback, LightGBM also receives validation-only `price_only` and `no_calendar` ablations. Pipelines fit median imputation/scaling only on the current training fold. Tree models receive training-only class weights to address label imbalance.

Expanding chronological validation intervals are 2023H1, 2023H2, 2024H1, and 2024H2, with the available 2018-onward history as training and purge by target resolution. Pooled out-of-fold macro-F1 chooses the model/feature specification for each lookback; pooled validation then chooses the overall lookback. Accuracy and log loss break ties. The selection is saved before the final evaluation.

The untouched test period is calendar 2025. Final estimators are refit on eligible 2018-2024 rows with labels resolved by test start. Every lookback's validation-selected specification is evaluated on the same test rows. Baselines are majority class, prior-regime persistence, prior-window momentum-rule, and a Laplace-smoothed Markov transition model. Accuracy, balanced accuracy, macro/weighted F1, log loss, multiclass Brier score, per-class precision/recall/F1, confusion matrices, and asset/quarter breakdowns are reported.

Paired moving-block bootstrap resamples seven-day UTC timestamp blocks and keeps both assets together; 2,000 repetitions quantify uncertainty in champion-minus-baseline metric differences. This addresses overlapping targets and cross-asset dependence better than row-wise iid intervals, but cannot establish future-market performance.

## Interpretation and limitations

BTC and ETH are correlated assets, not two independent markets. The test period is one calendar year. The operational target itself depends on fixed volatility and directional thresholds and is not ground truth. Regime imbalance, structural change, source-mirror errors, and exchange microstructure limit transfer. A test win is historical evidence only, not a promise to beat baselines in production, not a trading-profit result, and not financial advice. Do not tune thresholds, features, or models on the held-out test.
