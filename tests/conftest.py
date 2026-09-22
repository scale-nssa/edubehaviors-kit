"""Shared fixtures for the test suite."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def texts() -> list[str]:
    """A handful of classroom utterances, mixing questions and directives."""
    return [
        "Why do you think that is the answer?",
        "Please put your pencils down now.",
        "What do you notice about these two numbers?",
        "Everyone turn to page four.",
    ]


@pytest.fixture
def labelled_data() -> pd.DataFrame:
    """Forty labelled sentences across five transcripts, balanced over two labels.

    The two labels are separable from a handful of words alone, so a pipeline fitted on
    word features scores well above chance without any assertion model being downloaded.
    """
    rows = []
    for i in range(20):
        transcript = f"t{i % 5}"
        rows.append(
            {
                "sentence": f"Why do you think that is the answer, friend {i}?",
                "label": "question",
                "transcript": transcript,
            }
        )
        rows.append(
            {
                "sentence": f"Please put your pencils down now, group {i}.",
                "label": "directive",
                "transcript": transcript,
            }
        )
    return pd.DataFrame(rows)


class StubSetFitModel:
    """Stands in for a published SetFit model, so tests never reach the network.

    Scores a text as positive when it ends in a question mark, which is enough to exercise
    the annotator's device handling, batching and column naming.
    """

    def __init__(self) -> None:
        self.device: object | None = None
        # a SetFitModel references itself through its model card, so only the cycle collector
        # frees it; mirror that so tests catch a released model that is never collected
        self.model_card_data = SimpleNamespace(model=self)

    def to(self, device: object) -> StubSetFitModel:
        self.device = device
        return self

    def predict_proba(
        self, inputs: list[str], batch_size: int = 32, show_progress_bar: bool | None = None
    ) -> np.ndarray:
        positive = np.array([0.9 if text.strip().endswith("?") else 0.1 for text in inputs])
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace model loading with `StubSetFitModel`, keeping assertion tests offline."""
    from edubehaviors.annotation import AssertionAnnotator

    monkeypatch.setattr(AssertionAnnotator, "_load_model", lambda self, assertion: StubSetFitModel())


@pytest.fixture
def loaded_stub_models(monkeypatch: pytest.MonkeyPatch) -> list[weakref.ref[StubSetFitModel]]:
    """Replace model loading with `StubSetFitModel` like `stub_models`, recording each stub loaded.

    Returns:
        A weak reference to each stub, in load order, so tests can check which are still alive.
    """
    from edubehaviors.annotation import AssertionAnnotator

    loaded: list[weakref.ref[StubSetFitModel]] = []

    def load_model(annotator: AssertionAnnotator, assertion: str) -> StubSetFitModel:
        model = StubSetFitModel()
        loaded.append(weakref.ref(model))
        return model

    monkeypatch.setattr(AssertionAnnotator, "_load_model", load_model)
    return loaded
