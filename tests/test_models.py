import numpy as np

from regime_detection.models import make_model


def test_estimators_return_four_class_probabilities():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(220, 8))
    X[:20, 0] = np.nan
    y = np.tile(np.arange(4), 55)
    for name in ["lightgbm", "hist_gradient_boosting", "extra_trees", "logistic"]:
        model = make_model(name, seed=7, threads=1)
        model.fit(X, y)
        p = model.predict_proba(X[:7])
        assert p.shape == (7, 4)
        assert np.allclose(p.sum(axis=1), 1)
