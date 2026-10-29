# Expanded evaluation status and benchmark protocol

## Scope and evidence

The previously published `synthetic_metrics.json` is retained unchanged, including its `warning` and `test_data_note`. It is a **single synthetic smoke test** and is not evidence of live crypto performance. No real OHLCV source is available in this execution environment; Yahoo Finance/exchange data endpoints are not accessible here. Do not cite the old scores as BTC or crypto results.

This PR adds `experiments/robustness_benchmark.py` for the requested expanded evaluation. It generates separate trend-rich and mean-reverting OHLCV processes, runs four model families at each of four event horizons, uses three expanding walk-forward validation folds before the final 20% test, tunes probability thresholds from validation only, writes all-model test metrics, moving-block bootstrap intervals, model baselines, calibration/PR plots, permutation importance, current-state leakage audit, and the 3x3 label-definition sweep.

**The full repeated-seed benchmark has not completed in this run.** An exploratory execution was interrupted before writing its final artifacts, so no ≥10-seed aggregate or new calibration/importance plots are claimed here. Run the command below to create those reports before drawing conclusions. Its output directory is intentionally generated rather than committed; reported tables should be reviewed and then published with the exact `run_config.json`.

```bash
python -m pip install -r requirements.txt
python experiments/robustness_benchmark.py --seeds 10 --bars 900 --bootstrap 300 --outdir results/benchmark
```

## Evaluation details

- There are ten integer seeds for **each** generator (`trend_rich` and `mean_reverting`), with 900 hourly bars per generated series; distinct seeds are offset by 10,000 between generators.
- Each event label is 1 if the causal trailing range/trend state changes at least once during bars `t+1..t+H`; H is 4, 8, 12, or 16. The state uses trailing path efficiency and is independent of price direction.
- Walk-forward model/threshold selection uses three increasing validation windows (50–60%, 60–70%, 70–80%), with a 16-bar purge before each validation window. The final test is the last 20%; the final fit stops 16 eligible rows before the test boundary.
- Sixteen bars is sufficient for this target's maximum future reach: the latest training label can inspect state only through `t+16`, and the last training origin is at least 17 rows before the validation/test start. This purge prevents target windows crossing partitions. It does not remove serial dependence within partitions, hence 24-bar moving-block bootstrap intervals.
- Trend probability thresholds 0.30, 0.50, 0.70 and a validation-tuned threshold are reported. Tuning seeks precision ≥0.60 and recall ≥0.55 on out-of-fold validation probabilities; if unattainable, the fallback maximizes validation F1 without consulting test data. The validation goal is not guaranteed on test.
- `all_model_test_metrics.csv` reports all four model families, positive-event precision/recall/F1, PR-AUC (average precision), ROC-AUC, Brier score, log loss, and moving-block intervals for balanced accuracy, macro F1, and ROC-AUC. It also carries majority and no-shift persistence baselines; a model that fails to exceed persistence balanced accuracy must not be described as beating it.
- `current_regime_leakage_audit.csv` compares the current state predictor with and without path-efficiency, SMA-slope, range/ATR, and SMA-distance proxy features. This current state is a rule-generated target, not an objective market label.

## Interpretation and tradeability

Longer horizons have more opportunities for a state transition, which increases event prevalence and can change precision/recall; this is not automatically “horizon decay.” Feature limits, label ambiguity, and the event definition must be distinguished using the repeated-seed sweep and real-market data. Synthetic results alone cannot identify irreducible market predictability. The pipeline is **not tradeable as delivered**: it has no cost-aware execution simulation, slippage/fees, position sizing, or expected value per signal. Accuracy is not profitability.
