"""Multiclass metrics and time-block, paired uncertainty estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)

from .targets import CLASSES


def probabilities(model, X) -> np.ndarray:
    p = np.asarray(model.predict_proba(X), dtype=float)
    if p.shape != (len(X), 4) or not np.isfinite(p).all():
        raise ValueError("Classifier returned invalid probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities do not sum to one")
    return p


def score_predictions(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(4))
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=range(4))),
            "multiclass_brier": float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist()}


def chronological_folds(meta: pd.DataFrame, config: dict) -> list[dict]:
    """Split by decision time, pooled assets; purge labels crossing fold start."""
    decision = pd.to_datetime(meta["decision_time"], utc=True)
    end = pd.to_datetime(meta["target_end_time"], utc=True)
    start = pd.Timestamp(config["validation_start"], tz="UTC")
    folds = []
    for i, boundary in enumerate(config["validation_fold_ends"]):
        stop = pd.Timestamp(boundary, tz="UTC")
        train = np.flatnonzero((decision >= pd.Timestamp(config["train_start"], tz="UTC")) & (end <= start))
        valid = np.flatnonzero((decision >= start) & (decision < stop))
        if len(train) == 0 or len(valid) == 0:
            raise ValueError("Empty chronological fold")
        if end.iloc[train].max() > decision.iloc[valid].min():
            raise AssertionError("Training label reaches into validation")
        folds.append({"fold": i + 1, "train": train, "valid": valid,
                      "train_start": decision.iloc[train].min().isoformat(),
                      "train_end": decision.iloc[train].max().isoformat(),
                      "validation_start": start.isoformat(), "validation_end_exclusive": stop.isoformat()})
        start = stop
    return folds


def final_split(meta: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    decision = pd.to_datetime(meta["decision_time"], utc=True)
    end = pd.to_datetime(meta["target_end_time"], utc=True)
    test_start = pd.Timestamp(config["test_start"], tz="UTC")
    test_end = pd.Timestamp(config["test_end_exclusive"], tz="UTC")
    train = np.flatnonzero((decision >= pd.Timestamp(config["train_start"], tz="UTC")) & (end <= test_start))
    test = np.flatnonzero((decision >= test_start) & (decision < test_end))
    if end.iloc[train].max() > decision.iloc[test].min():
        raise AssertionError("Training label reaches into test")
    return train, test


def paired_block_bootstrap(y, candidate_p, baseline_p, decision_times, repetitions: int,
                           block: int, seed: int) -> dict:
    """Moving-block resampling of whole timestamps, retaining both assets together.

    A 42-bar (seven-day) block exceeds the six-bar label overlap. This quantifies
    sampling uncertainty on this test period, not uncertainty over all markets.
    """
    y = np.asarray(y, dtype=int)
    candidate = np.asarray(candidate_p).argmax(axis=1)
    baseline = np.asarray(baseline_p).argmax(axis=1)
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    unique = times.sort_values().unique()
    nblocks = int(np.ceil(len(unique) / block))
    rng = np.random.default_rng(seed)
    time_index = pd.Series(np.arange(len(unique)), index=unique)
    row_block = time_index.reindex(times).to_numpy()
    deltas = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(unique), size=nblocks)
        sampled = ((starts[:, None] + np.arange(block)[None, :]) % len(unique)).ravel()[:len(unique)]
        # Explicit multiplicity preserves blocks that are sampled more than once.
        multiplicity = np.bincount(sampled, minlength=len(unique))
        rows = np.repeat(np.arange(len(y)), multiplicity[row_block])
        deltas.append([f1_score(y[rows], candidate[rows], average="macro", zero_division=0) -
                       f1_score(y[rows], baseline[rows], average="macro", zero_division=0),
                       accuracy_score(y[rows], candidate[rows]) - accuracy_score(y[rows], baseline[rows]),
                       balanced_accuracy_score(y[rows], candidate[rows]) - balanced_accuracy_score(y[rows], baseline[rows])])
    arr = np.asarray(deltas)
    return {"method": "paired moving-block bootstrap over UTC decision timestamps; assets kept together",
            "repetitions": repetitions, "block_bars": block, "block_days": block / 6,
            "seed": seed, "metrics": {name: {"lower_95": float(np.quantile(arr[:, i], 0.025)),
                                             "upper_95": float(np.quantile(arr[:, i], 0.975)),
                                             "bootstrap_fraction_positive": float(np.mean(arr[:, i] > 0))}
                                      for i, name in enumerate(["macro_f1_delta", "accuracy_delta", "balanced_accuracy_delta"])}}
