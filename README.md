# Crypto 4H Regime Detection

A reproducible Python project for classifying the **next 24 hours** of BTCUSDT/ETHUSDT spot action from completed **4-hour candles**. It compares 12, 30, 60, 120, and 180-candle feature lookbacks using leakage-aware chronological validation and an untouched 2025 test year.

> **Held-out result:** [`results/test_results.md`](results/test_results.md) publishes metrics, predictions, model-selection records, and figures. The validation-selected 30-candle ExtraTrees model reached **0.316 macro-F1** on 2025 and beat the strongest test baseline by 5.36 macro-F1 points; it did **not** beat the majority/Markov baselines on raw accuracy (36.92% vs 39.98%). Results use a fixed forward regime definition and are not a profitability claim.

## Regimes

At a 4H close, compute the previous 42-return RMS volatility `sigma`, the next-six-candle log return `R`, and the next-six-candle return RMS `V`. Let `z = R / (sigma * sqrt(6))`:

- `bearish` when `z <= -0.5`
- `bullish` when `z >= +0.5`
- otherwise `volatile_range` when `V / sigma >= 1`
- otherwise `quiet_range`

This defines a practical forward classification target, not objective ground truth or a prediction of trading profit. Future values appear only in labels, never features.

## Setup and reproduction

Python 3.11+ is required. The exact tested dependencies are pinned.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

The pinned BTC/ETH 1-minute archives are downloaded from third-party GitHub mirrors by commit and SHA256-audited. Data is intentionally reconstructible rather than vendored; the committed `data/manifest.json` documents the source and processed parquet hashes. Downloading requires network access to GitHub/codeload.

```bash
# Build clean, complete 4H parquet and CSV files (archives cached outside Git).
python -m regime_detection prepare-data

# Re-run validation search, freeze selection, evaluate held-out test, save model/report.
python -m regime_detection train

# Or inspect/freeze the validation search without touching the final test.
python -m regime_detection train --stage validate --output results-validation

# Regenerate report/figures only from already saved evaluation artifacts.
python -m regime_detection report
```

The frozen experiment protocol is [`configs/experiment.json`](configs/experiment.json); pinned source commits are in [`configs/sources.json`](configs/sources.json). Once `results/protocol.sha256` exists, a changed protocol cannot silently overwrite that output directory; choose a fresh output path for a new study.

## Use the saved classifier

Input a CSV of UTC 4H candle **open** timestamps and the columns `open,high,low,close,volume,trade_count,taker_buy_volume`. Use only fully closed candles, ordered and unique, with 4H-aligned timestamps. `minute_count=240` is optional but, if present, incomplete rows are rejected.

```bash
python -m regime_detection predict --csv my_4h_candles.csv --output predictions.csv
```

By default, only the latest complete-lookback candle is emitted. Add `--all-rows` for all rows with sufficient history. Output probabilities, confidence, decision close, and the end of the 24H label horizon. The artifact is `results/models/regime_model.joblib`; only load trusted joblib files.

## Published artifacts

- [`results/test_results.md`](results/test_results.md): human-readable test report and lookback comparison.
- `results/test_metrics.csv`, `test_by_asset.csv`, `test_by_quarter.csv`: aggregate results.
- `results/test_predictions.csv`: every asset, lookback, truth, prediction and class probability.
- `results/validation_metrics.csv`, `validation_predictions.csv.gz`, `selection.json`: model search and frozen validation selection.
- `results/test_uncertainty.json`: paired moving-block bootstrap intervals.
- `results/lookback_comparison.png`, `confusion_matrix.png`: figures.
- `results/models/regime_model.joblib`: validation-champion model bundle.
- `data/manifest.json`: source/file/data-quality audit; raw prices remain reconstructible.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for data provenance, leakage controls, and caveats. This research software is not financial advice.
