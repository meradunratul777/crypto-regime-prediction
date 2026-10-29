# Test results: synthetic pipeline smoke test

**Data type:** synthetic OHLCV generated with NumPy (seed 2), 1,800 hourly bars. This is not BTC, ETH, or exchange data and these scores are not an estimate of live-market performance. It verifies the complete training, model selection, held-out evaluation, model save, and CSV export path. The synthetic series has no predictive guarantee and its metrics are included for transparency only.

**Setup:** trailing regime lookback 24 bars; path-efficiency threshold 0.35; chronological train/validation/test split with 16-bar embargo; 1,110 training rows, 266 validation rows, 353 held-out test rows. Four classifier families were compared independently per horizon. Validation balanced accuracy selected the model; test set was used only for final reporting.

| Forecast event | Selected model | Test accuracy | Test balanced accuracy | Test macro F1 | Test ROC-AUC | Shift-event precision | Shift-event recall | Test shift events |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| At least one regime change within 4 bars | Extra Trees | 81.59% | 72.94% | 0.688 | 0.871 | 0.408 | 0.608 | 51 / 353 |
| At least one regime change within 8 bars | Logistic regression | 76.77% | 71.05% | 0.701 | 0.744 | 0.525 | 0.598 | 87 / 353 |
| At least one regime change within 12 bars | Logistic regression | 71.95% | 68.85% | 0.688 | 0.708 | 0.587 | 0.592 | 120 / 353 |
| At least one regime change within 16 bars | Logistic regression | 67.42% | 65.86% | 0.660 | 0.674 | 0.615 | 0.568 | 146 / 353 |

Shift-event metrics use the positive (`shift`) class. Accuracy falls as the forecast window expands, consistent with more opportunities for a state transition, but this pattern is specific to this test series. The current state classifier scored 96.32% test accuracy and 97.85% balanced accuracy; its target is defined by a rolling price-efficiency rule, so this is agreement with that proxy—not proof that the market has an objectively known regime.

## Reproduce

Install `requirements.txt`, generate synthetic data with a fixed seed, and run:

```bash
python main.py --csv your_ohlcv.csv --interval 1h --horizon 24 --output artifacts
```

The exact predictions and complete metrics from each run are written to `artifacts/test_predictions.csv` and `artifacts/metrics.json`. The synthetic fixture used for the report was generated locally and is intentionally not committed. To publish meaningful crypto results, rerun on an identified real OHLCV source and period, then replace this report with the data source, sample dates, data-cleaning details, and untouched test metrics.
