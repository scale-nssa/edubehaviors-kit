"""Smoke tests for the classifier factory."""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.validation import check_is_fitted

from edubehaviors import standard_classifier
from edubehaviors.models import DEFAULT_C_GRID


def test_returns_an_unfitted_classifier():
    classifier = standard_classifier()
    assert isinstance(classifier, LogisticRegressionCV)
    assert not hasattr(classifier, "coef_")


def test_applies_the_documented_defaults():
    classifier = standard_classifier()
    assert classifier.Cs == DEFAULT_C_GRID
    assert classifier.scoring == "f1_macro"
    assert classifier.class_weight == "balanced"
    assert classifier.cv == 5


def test_default_grid_spans_the_documented_range():
    assert len(DEFAULT_C_GRID) == 12
    assert DEFAULT_C_GRID[0] == 1e-3
    assert DEFAULT_C_GRID[-1] == 1e2


def test_overrides_are_forwarded():
    classifier = standard_classifier(Cs=3, scoring="accuracy", class_weight=None, cv=2, max_iter=50)
    assert classifier.Cs == 3
    assert classifier.scoring == "accuracy"
    assert classifier.class_weight is None
    assert classifier.cv == 2
    assert classifier.max_iter == 50


def test_fits_on_a_separable_problem():
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.normal(-2.0, 0.5, (20, 2)), rng.normal(2.0, 0.5, (20, 2))])
    outcome = np.array([0] * 20 + [1] * 20)
    classifier = standard_classifier(Cs=3, cv=2, random_state=0).fit(features, outcome)
    check_is_fitted(classifier)
    assert classifier.score(features, outcome) > 0.9
