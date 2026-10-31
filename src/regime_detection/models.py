"""Small, predeclared model-search space; all fitting confined to training folds."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["lightgbm", "hist_gradient_boosting", "extra_trees", "logistic"]


def make_model(name: str, seed: int = 1729, threads: int = 2):
    if name == "lightgbm":
        return LGBMClassifier(objective="multiclass", num_class=4, n_estimators=400,
                              learning_rate=0.035, num_leaves=15, max_depth=5,
                              min_child_samples=80, reg_lambda=10.0, reg_alpha=0.5,
                              colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                              class_weight="balanced", random_state=seed,
                              n_jobs=threads, verbosity=-1, deterministic=True,
                              force_col_wise=True)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=80,
                                               l2_regularization=5.0, early_stopping=False,
                                               class_weight="balanced", random_state=seed)
    if name == "extra_trees":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20,
                                                        max_features=0.7, class_weight="balanced",
                                                        n_jobs=threads, random_state=seed))])
    if name == "logistic":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=0.03, class_weight="balanced",
                                                       max_iter=1500, random_state=seed))])
    raise ValueError(f"Unknown model {name}")


class MajorityBaseline:
    def fit(self, X, y):
        self.classes_ = np.arange(4)
        counts = np.bincount(np.asarray(y, dtype=int), minlength=4)
        self.majority = int(counts.argmax())
        self.probabilities = counts / counts.sum()
        return self

    def predict(self, X):
        return np.full(len(X), self.majority, dtype=int)

    def predict_proba(self, X):
        return np.tile(self.probabilities, (len(X), 1))


class PersistenceBaseline:
    """The preceding completed 24H regime, already known at the decision close."""
    def __init__(self, previous):
        self.previous = np.asarray(previous, dtype=int)

    def predict(self, X):
        return self.previous

    def predict_proba(self, X):
        return np.eye(4)[self.previous]


class MomentumBaseline:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction, dtype=int)

    def predict(self, X):
        return self.prediction

    def predict_proba(self, X):
        return np.eye(4)[self.prediction]


class MarkovBaseline:
    """Train-only transition matrix conditioned on the last completed regime."""
    def fit(self, previous, y):
        counts = np.ones((4, 4))  # Laplace smoothing
        np.add.at(counts, (np.asarray(previous, dtype=int), np.asarray(y, dtype=int)), 1)
        self.matrix = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, previous):
        return self.matrix[np.asarray(previous, dtype=int)]

    def predict(self, previous):
        return self.predict_proba(previous).argmax(axis=1)
