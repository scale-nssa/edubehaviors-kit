"""The baseline classifier `ClassificationPipeline` fits on annotated features."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

DEFAULT_C_GRID = tuple(float(c) for c in np.logspace(-3, 2, 12))
"""Twelve log-spaced inverse regularization strengths, from 1e-3 to 1e2."""


def standard_classifier(
    *,
    Cs: int | Sequence[float] = DEFAULT_C_GRID,
    scoring: str | Callable = "f1_macro",
    class_weight: Mapping | Literal["balanced"] | None = "balanced",
    cv: int = 5,
    **kwargs: Any,
) -> LogisticRegressionCV:
    """Factory method for customized sklearn.linear_model.LogisticRegressionCV.

    Args:
        Cs: The inverse regularization strengths to choose between, or how many to space
            logarithmically between 1e-4 and 1e4.
        scoring: The metric the penalty is chosen against, as a scorer name or a callable.
        class_weight: How much each class counts toward the loss.
        cv: The number of StratifiedKFold folds used.
        **kwargs: Keyword arguments to pass to LogisticRegressionCV.

    Returns:
        An unfitted `LogisticRegressionCV`.
    """
    return LogisticRegressionCV(Cs=Cs, scoring=scoring, class_weight=class_weight, cv=cv, **kwargs)
