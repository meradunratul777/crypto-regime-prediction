"""Generate the published report and figures exclusively from saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .targets import CLASSES


def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free Markdown tables (avoids optional tabulate)."""
    columns = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def create_report(output: Path, data_dir: Path, config: dict) -> None:
    manifest = json.loads((output / "run_manifest.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    test = pd.read_csv(output / "test_metrics.csv")
    assets = pd.read_csv(output / "test_by_asset.csv")
    validation = pd.read_csv(output / "validation_metrics.csv")
    best = manifest["selected_best"]
    lb = int(best["lookback"])
    champion = test[(test.lookback == lb) & (test.model == best["model"]) & (test.feature_set == best["feature_set"])].iloc[0]
    persistence = test[test.model == "persistence"].iloc[0]
    best_baseline = test[test.feature_set == "rule"].sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    delta = champion.macro_f1 - best_baseline.macro_f1
    verdict = "beats" if delta > 0 else "does not beat"
    lines = ["# Crypto 4H regime detection — held-out test results", "",
             "## Outcome", "",
             f"The validation-selected model is **{best['model']}** with **{lb} completed 4H candles ({lb/6:g} days)** of lookback "
             f"and **{best['feature_set']}** features. It {verdict} the strongest test baseline by "
             f"**{delta*100:+.2f} macro-F1 percentage points**. Selection used validation data only; the test baseline ranking is reported for comparison, not used to choose the model.", "",
             f"**Important accuracy caveat:** champion accuracy is {champion.accuracy:.2%}, below the majority/Markov baselines at {test[test.model.isin(['majority', 'markov'])].accuracy.max():.2%}. "
             "It improves macro-F1 and balanced accuracy, but does not beat those simple baselines on raw accuracy or log loss; the paired accuracy interval versus persistence includes zero. "
             "The fixed selection objective was validation macro-F1 to avoid choosing a model that ignores minority regimes. Results are reported without test-driven tuning.", "", 
             f"- **Test accuracy:** {champion.accuracy:.2%}",
             f"- **Balanced accuracy:** {champion.balanced_accuracy:.2%}",
             f"- **Macro-F1:** {champion.macro_f1:.4f}",
             f"- **Log loss:** {champion.log_loss:.4f}",
             f"- **Multiclass Brier score:** {champion.multiclass_brier:.4f}",
             f"- **Test rows:** {int(champion.n):,} pooled BTC/ETH decisions ({manifest['test_n']//2:,} per asset)",
             f"- **Decision period:** {manifest['test_decision_start']} through {manifest['test_decision_end']}",
             f"- **Final label resolves:** {manifest['test_label_end_max']}", "",
             "## What is being detected?", "",
             "At every 4-hour close, classify the **next 24 hours** into one of four fixed, mutually exclusive regimes. "
             "This is forward regime classification, not a claim to know an unobservable 'true' market state or a price/profit forecast.", "",
             "Let `sigma` be RMS of the previous 42 one-candle log returns, `R` the next-six-candle net log return, "
             "and `V` the RMS of those six future returns. With `z = R / (sigma * sqrt(6))`:", "",
             "- **bearish:** z <= -0.5", "- **bullish:** z >= +0.5 (checked after bearish; boundaries are disjoint)",
             "- **volatile_range:** neither directional condition, and V/sigma >= 1",
             "- **quiet_range:** neither directional condition, and V/sigma < 1", "",
             "The label uses future prices only to construct the training/evaluation target. Those future quantities are never model inputs. "
             "Thresholds and the 24-hour horizon are fixed across all lookbacks, so the comparison uses identical targets and rows.", "",
             "## Lookback comparison — untouched 2025 test year", "",
             "Each row refits that lookback's validation-selected specification on 2018–2024. The overall champion was chosen before test evaluation."]
    table = test[test.feature_set != "rule"].sort_values("lookback").copy()
    table["lookback_days"] = table.lookback / 6
    table["selected_model"] = table.model + " / " + table.feature_set
    display = table[["lookback", "lookback_days", "selected_model", "n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]].copy()
    for col in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display["lookback_days"] = display.lookback_days.map(lambda x: f"{x:g}")
    lines += [markdown_table(display), "", "## Baselines — identical test rows", ""]
    base = test[test.feature_set == "rule"].copy()
    display = base[["model", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]].copy()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    lines += [markdown_table(display), "",
              "`majority` learns the training class mode. `persistence` repeats the immediately preceding completed 24H regime. "
              "`momentum_rule` applies the target's economic rule to the preceding window. `markov` learns a smoothed 4×4 transition matrix on training data only. "
              "Logistic regression is also included in the model search as a regularized linear learned baseline.", "",
              "## Per-asset champion results", ""]
    asset = assets[(assets.lookback == lb) & (assets.model == best["model"]) & (assets.feature_set == best["feature_set"])].copy()
    display = asset[["symbol", "n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]].copy()
    for col in display.columns[2:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    lines += [markdown_table(display), "", "## Champion per-class results", ""]
    per_class = pd.DataFrame(manifest["per_class_champion"]).T.reset_index(names="regime")
    per_class["support"] = per_class.support.astype(int)
    for col in ["precision", "recall", "f1"]:
        per_class[col] = per_class[col].map(lambda x: f"{x:.4f}")
    lines += [markdown_table(per_class), "", "Confusion matrix (rows actual, columns predicted; order: " + ", ".join(CLASSES) + "):", "", "```text",
              str(np.asarray(manifest["confusion_matrix_champion"])), "```", "", "## Uncertainty: paired time-block bootstrap", "",
              "Whole decision timestamps are resampled in seven-day moving blocks, keeping BTC and ETH together. "
              "This preserves cross-asset dependence and exceeds the six-candle label overlap. These intervals describe this historical sample, not a guarantee over future market regimes.", ""]
    for baseline, result in manifest["uncertainty"].items():
        lines.append(f"Versus **{baseline}** ({result['repetitions']:,} repetitions):")
        for metric, values in result["metrics"].items():
            lines.append(f"- {metric}: 95% interval [{values['lower_95']:+.4f}, {values['upper_95']:+.4f}]")
        lines.append("")
    lines += ["## Data and leakage controls", "",
              "- Real Binance-spot BTCUSDT and ETHUSDT one-minute candles from pinned third-party GitHub archives; this is not an exchange-certified dataset, and the source repositories do not report a data license.",
              "- Raw archives/processed price files are not redistributed; review third-party source terms before reuse.",
              "- Source commits, monthly CSV hashes, archive hashes and processed-data hashes: [`data/manifest.json`](../data/manifest.json).",
              "- 240 distinct, valid, aligned minutes required per 4H candle; OHLC aggregated correctly and volumes/counts summed. No filled candles.",
              "- Invalid/padding records and incomplete candles rejected. Missing grid slots remain missing; rolling windows never bridge them silently.",
              "- Asset-local, right-aligned features; no raw price level, centered rolling window, global normalization, future return or target-derived feature.",
              "- Lookbacks include their indicator warm-up and nested windows within the stated candle budget. All lookbacks use the same eligible rows.",
              "- Expanding validation folds: 2023H1, 2023H2, 2024H1, 2024H2. Training labels must resolve before each validation fold begins.",
              "- Final training includes 2018–2024, purged so no label reaches into the test decision period. No random train/test split.",
              "- Model family, feature ablation and lookback selected by pooled validation macro-F1; accuracy then negative log loss break ties.",
              "- Median imputation/scaling (where needed) fitted inside each training fold. BTC and ETH are pooled only after causal feature construction.",
              "- Test is evaluated after selection is frozen. All lookback test results are published, not just the winner.", "",
              "## Validation search", "",
              f"{len(selection['candidates'])} specifications; four expanding folds each; {len(config['lookbacks'])} lookbacks. "
              "Full fold metrics and out-of-fold predictions are saved. The table shows each lookback's selected specification."]
    val_table = pd.DataFrame(selection["selected_per_lookback"]).T
    val_table = val_table[["lookback", "model", "feature_set", "feature_count", "validation_macro_f1", "validation_accuracy"]]
    val_table = val_table.sort_values("lookback")
    for col in ["validation_macro_f1", "validation_accuracy"]:
        val_table[col] = val_table[col].astype(float).map(lambda x: f"{x:.4f}")
    lines += [markdown_table(val_table), "", "## Figures", "",
              "![Lookback comparison](lookback_comparison.png)", "", "![Champion confusion matrix](confusion_matrix.png)", "",
              "## Reproducibility and limits", "",
              "See the [README](../README.md) and [methodology](../docs/METHODOLOGY.md). Run `python -m regime_detection train` to rebuild data and reproduce the experiment, "
              "or `python -m regime_detection report` to regenerate this report from saved metrics. "
              "The trained validation-champion model is `results/models/regime_model.joblib`.", "",
              "This benchmark classifies an explicit operational regime definition on two correlated assets. It is not independent cross-market validation, "
              "not proof of causal market-state discovery, and not a trading-profit backtest. Volatile range is a minority class; macro-F1 and per-class recall are therefore important. "
              "A favorable historical test does not guarantee future accuracy. No result here was selected or tuned using the test set.", "",
              f"Environment: Python {manifest['environment']['python']}, scikit-learn {manifest['environment']['scikit_learn']}, "
              f"LightGBM {manifest['environment']['lightgbm']}, pandas {manifest['environment']['pandas']}, NumPy {manifest['environment']['numpy']}.", ""]
    (output / "test_results.md").write_text("\n".join(lines))
    # Summary CSV convenient for downstream comparison.
    summary = test.copy()
    summary["lookback_days"] = summary.lookback / 6
    summary.to_csv(output / "results_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    models = table.sort_values("lookback")
    for ax, metric, title in [(axes[0], "macro_f1", "Macro-F1 (selection metric)"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(models.lookback, models[metric], "o-", color="#2878b5", label="Selected ML per lookback")
        for name, color in [("persistence", "#d95f02"), ("markov", "#7570b3"), ("momentum_rule", "#66a61e")]:
            value = test.loc[test.model == name, metric].iloc[0]
            ax.axhline(value, ls="--", lw=1, color=color, label=name)
        ax.set(xlabel="Lookback (completed 4H candles)", ylabel=metric, title=title)
        ax.set_xticks(config["lookbacks"])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("2025 held-out test: BTCUSDT + ETHUSDT, next-24H regime")
    fig.savefig(output / "lookback_comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    cm = np.asarray(manifest["confusion_matrix_champion"])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title=f"Champion: {lb}-bar lookback (row-normalized)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{normalized[i,j]:.2f}\n(n={cm[i,j]})", ha="center", va="center",
                    color="white" if normalized[i,j] > .5 else "#172b4d", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output / "confusion_matrix.png", dpi=170)
    plt.close(fig)
