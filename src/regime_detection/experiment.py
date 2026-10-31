"""Reproducible lookback/model search, followed by one final held-out evaluation."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from .data import sha256_file
from .evaluation import (chronological_folds, final_split, paired_block_bootstrap,
                         probabilities, score_predictions)
from .features import feature_group_ablation, make_features
from .models import (MajorityBaseline, MarkovBaseline, MomentumBaseline,
                     PersistenceBaseline, make_model)
from .targets import CLASSES, momentum_baseline, regime_targets

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier"]


def _metric_row(metrics: dict, **labels) -> dict:
    return {**labels, "n": metrics["n"], **{key: metrics[key] for key in METRICS}}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")




def build_dataset(data_dir: Path, config: dict):
    frames, metadata, targets, previous, momentum = {}, [], [], [], []
    max_lookback = max(config["lookbacks"])
    for symbol in config["symbols"]:
        bars = pd.read_parquet(data_dir / f"{symbol}_4h.parquet")
        target = regime_targets(bars, config["horizon"], config["label_vol_window"],
                                config["direction_threshold"], config["range_vol_threshold"])
        rule = momentum_baseline(bars, target, config["horizon"], config["direction_threshold"],
                                 config["range_vol_threshold"])
        feature_by_lookback = {lb: make_features(bars, lb) for lb in config["lookbacks"]}
        eligible = (feature_by_lookback[max_lookback].notna().any(axis=1) & target["target"].notna()
                    & target["previous_regime"].notna() & rule.notna())
        meta = target.loc[eligible, ["decision_time", "target_end_time"]].copy()
        meta.insert(0, "symbol", symbol)
        meta["candle_open_time"] = meta.index
        metadata.append(meta.reset_index(drop=True))
        targets.append(target.loc[eligible, "target"].astype(int).reset_index(drop=True))
        previous.append(target.loc[eligible, "previous_regime"].astype(int).reset_index(drop=True))
        momentum.append(rule.loc[eligible].astype(int).reset_index(drop=True))
        for lookback, x in feature_by_lookback.items():
            x = x.loc[eligible].copy()
            x.insert(0, "symbol", symbol)
            frames.setdefault(lookback, []).append(x.reset_index(drop=True))
    X = {lb: pd.concat(parts, ignore_index=True) for lb, parts in frames.items()}
    meta = pd.concat(metadata, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    prev = pd.concat(previous, ignore_index=True)
    mom = pd.concat(momentum, ignore_index=True)
    if not meta["decision_time"].is_monotonic_increasing:
        order = meta.sort_values(["decision_time", "symbol"], kind="stable").index
        meta, y, prev, mom = (frame.loc[order].reset_index(drop=True) for frame in (meta, y, prev, mom))
        X = {lb: frame.loc[order].reset_index(drop=True) for lb, frame in X.items()}
    return X, meta, y, prev, mom


def candidate_specs(config: dict) -> list[dict]:
    specs = [{"model": name, "feature_set": "full"} for name in config["models"]]
    # Predeclared feature-engineering ablations, not selected using test scores.
    specs += [{"model": "lightgbm", "feature_set": "price_only"},
              {"model": "lightgbm", "feature_set": "no_calendar"}]
    return specs


def _baselines(train_idx, eval_idx, y, previous, momentum):
    x_train = np.zeros((len(train_idx), 1))
    x_eval = np.zeros((len(eval_idx), 1))
    majority = MajorityBaseline().fit(x_train, y.iloc[train_idx])
    markov = MarkovBaseline().fit(previous.iloc[train_idx], y.iloc[train_idx])
    return {"majority": majority.predict_proba(x_eval),
            "persistence": PersistenceBaseline(previous.iloc[eval_idx]).predict_proba(x_eval),
            "momentum_rule": MomentumBaseline(momentum.iloc[eval_idx]).predict_proba(x_eval),
            "markov": markov.predict_proba(previous.iloc[eval_idx])}


def run_validation(X, meta, y, previous, momentum, config, output: Path, threads: int) -> dict:
    folds = chronological_folds(meta, config)
    records, oof_records = [], []
    baseline_oof = {name: np.full((len(y), 4), np.nan) for name in ["majority", "persistence", "momentum_rule", "markov"]}
    for fold in folds:
        for name, p in _baselines(fold["train"], fold["valid"], y, previous, momentum).items():
            baseline_oof[name][fold["valid"]] = p
    for name, p in baseline_oof.items():
        mask = np.isfinite(p).all(axis=1)
        records.append(_metric_row(score_predictions(y.loc[mask], p[mask]), lookback=0, model=name,
                                   feature_set="rule", fold=0, split="validation_oof", train_n=0, validation_n=int(mask.sum())))
    selections = {}
    for lookback in config["lookbacks"]:
        frame = X[lookback]
        columns = [c for c in frame.columns if c != "symbol"]
        for spec in candidate_specs(config):
            selected_columns = feature_group_ablation(columns, spec["feature_set"])
            started = time.monotonic()
            oof = np.full((len(y), 4), np.nan)
            for fold in folds:
                train, valid = fold["train"], fold["valid"]
                model = make_model(spec["model"], config["seed"], threads)
                model.fit(frame.iloc[train][selected_columns], y.iloc[train])
                p = probabilities(model, frame.iloc[valid][selected_columns])
                oof[valid] = p
                records.append(_metric_row(score_predictions(y.iloc[valid], p), lookback=lookback,
                                           **spec, fold=fold["fold"], split="validation",
                                           train_n=len(train), validation_n=len(valid)))
                for name, p_base in _baselines(train, valid, y, previous, momentum).items():
                    records.append(_metric_row(score_predictions(y.iloc[valid], p_base), lookback=lookback,
                                               model=name, feature_set="rule", fold=fold["fold"], split="validation",
                                               train_n=len(train), validation_n=len(valid)))
            mask = np.isfinite(oof).all(axis=1)
            aggregate = score_predictions(y.loc[mask], oof[mask])
            records.append(_metric_row(aggregate, lookback=lookback, **spec, fold=0, split="validation_oof",
                                       train_n=0, validation_n=int(mask.sum())))
            selections[f"{lookback}:{spec['model']}:{spec['feature_set']}"] = {
                "lookback": lookback, **spec, "validation_macro_f1": aggregate["macro_f1"],
                "validation_accuracy": aggregate["accuracy"], "validation_balanced_accuracy": aggregate["balanced_accuracy"],
                "validation_log_loss": aggregate["log_loss"],
                "mean_fold_macro_f1": float(np.mean([r["macro_f1"] for r in records if r["lookback"] == lookback and r["model"] == spec["model"]
                                                     and r["feature_set"] == spec["feature_set"] and r["split"] == "validation"])),
                "feature_count": len(selected_columns), "elapsed_seconds": time.monotonic() - started}
            valid_meta = meta.loc[mask].copy()
            valid_meta["lookback"] = lookback
            valid_meta["model"] = spec["model"]
            valid_meta["feature_set"] = spec["feature_set"]
            valid_meta["actual"] = y.loc[mask].map(dict(enumerate(CLASSES)))
            valid_meta["predicted"] = np.asarray(CLASSES)[oof[mask].argmax(axis=1)]
            for i, name in enumerate(CLASSES):
                valid_meta[f"prob_{name}"] = oof[mask, i]
            oof_records.append(valid_meta)
            print(f"VALIDATION L={lookback:3d} {spec['model']:24s} {spec['feature_set']:11s} "
                  f"macroF1={aggregate['macro_f1']:.4f} accuracy={aggregate['accuracy']:.4f} "
                  f"({time.monotonic()-started:.1f}s)", flush=True)
    pd.DataFrame(records).to_csv(output / "validation_metrics.csv", index=False)
    pd.concat(oof_records, ignore_index=True).to_csv(output / "validation_predictions.csv.gz", index=False, compression="gzip")
    winners = {}
    for lookback in config["lookbacks"]:
        candidates = [s for s in selections.values() if s["lookback"] == lookback]
        winners[str(lookback)] = max(candidates, key=lambda s: (s["validation_macro_f1"], s["validation_accuracy"], -s["validation_log_loss"]))
    best = max(winners.values(), key=lambda s: (s["validation_macro_f1"], s["validation_accuracy"], -s["validation_log_loss"]))
    result = {"protocol": "Expanding 2023H1/2023H2/2024H1/2024H2 validation; pooled OOF macro-F1 selects model/features; then lookback. No test scores used.",
              "folds": [{k: v for k, v in f.items() if k not in ("train", "valid")} | {"train_n": len(f["train"]), "validation_n": len(f["valid"])} for f in folds],
              "candidates": list(selections.values()), "selected_per_lookback": winners, "selected_best": best}
    write_json(output / "selection.json", result)
    return result


def run_test(X, meta, y, previous, momentum, config, output: Path, selection: dict, threads: int) -> dict:
    train, test = final_split(meta, config)
    base_prob = _baselines(train, test, y, previous, momentum)
    test_records, asset_records, quarter_records, prediction_frames, models = [], [], [], [], {}
    for name, p in base_prob.items():
        metrics = score_predictions(y.iloc[test], p)
        test_records.append(_metric_row(metrics, lookback=0, model=name, feature_set="rule", split="test"))
        for symbol in config["symbols"]:
            mask = meta.iloc[test]["symbol"].eq(symbol).to_numpy()
            asset_records.append(_metric_row(score_predictions(y.iloc[test][mask], p[mask]), lookback=0,
                                             model=name, feature_set="rule", symbol=symbol, split="test"))
    chosen = selection["selected_best"]
    for lookback in config["lookbacks"]:
        spec = selection["selected_per_lookback"][str(lookback)]
        columns = feature_group_ablation([c for c in X[lookback].columns if c != "symbol"], spec["feature_set"])
        model = make_model(spec["model"], config["seed"], threads)
        model.fit(X[lookback].iloc[train][columns], y.iloc[train])
        p = probabilities(model, X[lookback].iloc[test][columns])
        metrics = score_predictions(y.iloc[test], p)
        labels = {"lookback": lookback, "model": spec["model"], "feature_set": spec["feature_set"], "split": "test"}
        test_records.append(_metric_row(metrics, **labels))
        for symbol in config["symbols"]:
            mask = meta.iloc[test]["symbol"].eq(symbol).to_numpy()
            asset_records.append(_metric_row(score_predictions(y.iloc[test][mask], p[mask]), symbol=symbol, **labels))
        quarters = pd.to_datetime(meta.iloc[test]["decision_time"], utc=True).dt.to_period("Q").astype(str).to_numpy()
        for quarter in sorted(set(quarters)):
            mask = quarters == quarter
            quarter_records.append(_metric_row(score_predictions(y.iloc[test][mask], p[mask]), quarter=quarter, **labels))
        pred = meta.iloc[test][["symbol", "candle_open_time", "decision_time", "target_end_time"]].copy()
        pred["lookback"] = lookback
        pred["model"] = spec["model"]
        pred["feature_set"] = spec["feature_set"]
        pred["actual"] = y.iloc[test].map(dict(enumerate(CLASSES))).to_numpy()
        pred["predicted"] = np.asarray(CLASSES)[p.argmax(axis=1)]
        for i, name in enumerate(CLASSES):
            pred[f"prob_{name}"] = p[:, i]
        pred["persistence_prediction"] = np.asarray(CLASSES)[base_prob["persistence"].argmax(axis=1)]
        prediction_frames.append(pred)
        models[lookback] = (model, columns, p, metrics)
        print(f"TEST L={lookback:3d} {spec['model']:24s} macroF1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
    test_frame = pd.DataFrame(test_records)
    test_frame.to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(asset_records).to_csv(output / "test_by_asset.csv", index=False)
    pd.DataFrame(quarter_records).to_csv(output / "test_by_quarter.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "test_predictions.csv", index=False)
    # The strongest validation baseline is chosen without inspecting test results.
    val = pd.read_csv(output / "validation_metrics.csv")
    val_baselines = val[(val.split == "validation") & (val.feature_set == "rule")].groupby("model").macro_f1.mean()
    strongest_baseline = str(val_baselines.idxmax())
    model, columns, p, metrics = models[chosen["lookback"]]
    uncertainty = {}
    for baseline in dict.fromkeys(["persistence", strongest_baseline]):
        uncertainty[baseline] = paired_block_bootstrap(y.iloc[test], p, base_prob[baseline],
                                                       meta.iloc[test]["decision_time"], config["bootstrap_repetitions"],
                                                       config["bootstrap_block_bars"], config["seed"])
    write_json(output / "test_uncertainty.json", {"selected_lookback": chosen["lookback"], "comparisons": uncertainty})
    # Save only the validation-champion artifact; all lookback results remain auditable.
    bundle = {"model": model, "feature_columns": columns, "lookback": chosen["lookback"],
              "classes": CLASSES, "config": config, "schema_version": 1,
              "target": "Next 6 completed 4H candles (24 hours), four fixed regimes",
              "required_columns": ["open", "high", "low", "close", "volume", "trade_count", "taker_buy_volume"]}
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, model_dir / "regime_model.joblib", compress=3)
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": columns, "importance": estimator.feature_importances_})
        importance.sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    details = {"selected_best": chosen, "selected_per_lookback": selection["selected_per_lookback"],
               "train_n": len(train), "test_n": len(test),
               "train_decision_start": meta.iloc[train]["decision_time"].min(),
               "train_decision_end": meta.iloc[train]["decision_time"].max(),
               "train_label_end_max": meta.iloc[train]["target_end_time"].max(),
               "test_decision_start": meta.iloc[test]["decision_time"].min(),
               "test_decision_end": meta.iloc[test]["decision_time"].max(),
               "test_label_end_max": meta.iloc[test]["target_end_time"].max(),
               "test_class_counts": {CLASSES[k]: int(v) for k, v in y.iloc[test].value_counts().sort_index().items()},
               "strongest_validation_baseline": strongest_baseline,
               "test_metrics": test_records, "per_class_champion": metrics["per_class"],
               "confusion_matrix_champion": metrics["confusion_matrix"],
               "uncertainty": uncertainty,
               "model_sha256": sha256_file(model_dir / "regime_model.joblib"),
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__}}
    write_json(output / "run_manifest.json", details)
    return details


def run_experiment(data_dir: Path, config_path: Path, output: Path, threads: int = 2, stage: str = "all") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    manifest_hash = sha256_file(data_dir / "manifest.json")
    protocol_hash = f"config={config_hash}\ndata_manifest={manifest_hash}"
    protocol_path = output / "protocol.sha256"
    if protocol_path.exists() and protocol_path.read_text().strip() != protocol_hash:
        raise ValueError("Protocol or data manifest changed. Use a fresh output directory; do not overwrite a prior experiment.")
    protocol_path.write_text(protocol_hash + "\n")
    X, meta, y, previous, momentum = build_dataset(data_dir, config)
    print(f"Dataset: {len(y):,} pooled rows; {len(X[max(config['lookbacks'])].columns)-1} maximum features; "
          f"{meta.decision_time.min()} to {meta.decision_time.max()}", flush=True)
    if stage == "validate":
        return run_validation(X, meta, y, previous, momentum, config, output, threads)
    selection_path = output / "selection.json"
    if selection_path.exists():
        print("Reusing frozen validation selection; test remains untouched until now.", flush=True)
        selection = json.loads(selection_path.read_text())
    else:
        selection = run_validation(X, meta, y, previous, momentum, config, output, threads)
        # Freeze selection to disk before any test prediction is computed.
        selection = json.loads(selection_path.read_text())
    result = run_test(X, meta, y, previous, momentum, config, output, selection, threads)
    from .report import create_report
    create_report(output, data_dir, config)
    return result
