import numpy as np
import pandas as pd

from regime_detection.evaluation import chronological_folds, final_split, paired_block_bootstrap


def sample_meta():
    rows = []
    for symbol in ["BTC", "ETH"]:
        for t in pd.date_range("2018-02-01", "2025-12-31", freq="4h", tz="UTC"):
            rows.append((symbol, t, t + pd.Timedelta(hours=24)))
    return pd.DataFrame(rows, columns=["symbol", "decision_time", "target_end_time"])


def test_time_splits_are_purged_and_chronological():
    meta = sample_meta()
    config = {"train_start": "2018-02-01", "validation_start": "2023-01-01",
              "validation_fold_ends": ["2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"],
              "test_start": "2025-01-01", "test_end_exclusive": "2026-01-01"}
    folds = chronological_folds(meta, config)
    for fold in folds:
        assert meta.iloc[fold["train"]].target_end_time.max() <= meta.iloc[fold["valid"]].decision_time.min()
    train, test = final_split(meta, config)
    assert meta.iloc[train].target_end_time.max() <= meta.iloc[test].decision_time.min()
    assert meta.iloc[test].decision_time.min() >= pd.Timestamp("2025-01-01", tz="UTC")


def test_paired_bootstrap_keeps_shape_and_finite_intervals():
    n = 120
    y = np.tile([0, 1, 2, 3], 30)
    candidate = np.eye(4)[y]
    baseline = np.roll(candidate, 1, axis=1)
    times = np.repeat(pd.date_range("2025-01-01", periods=60, freq="4h", tz="UTC"), 2)
    result = paired_block_bootstrap(y, candidate, baseline, times, 100, 6, 42)
    assert result["repetitions"] == 100
    for values in result["metrics"].values():
        assert np.isfinite(values["lower_95"])
        assert np.isfinite(values["upper_95"])
