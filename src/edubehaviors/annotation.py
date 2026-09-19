"""Annotators that score text against the EduBehaviors assertions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Literal, get_args

import pandas as pd
import torch
from sentence_transformers.util import get_device_name
from setfit import SetFitModel
from tqdm import tqdm

from .constants import ALL, DEFAULT_WORDS, EXISTING_ASSERTIONS, Assertion

WordMethod = Literal["count", "contains"]
EXISTING_WORD_METHODS: tuple[WordMethod, ...] = get_args(WordMethod)
"""The matching methods `WordAnnotator` supports."""


def _prepare_inputs(inputs: list[str] | pd.Series, *, allow_na: bool = False) -> pd.Series:
    """Coerce inputs to a Series of strings.

    Args:
        inputs: The texts to annotate.
        allow_na: Whether missing values are allowed. When they are, the Series is cast to
            the nullable string dtype, so that missing values propagate as `pd.NA` rather
            than being reported as zero occurrences.

    Returns:
        The inputs as a Series, keeping their original index.

    Raises:
        TypeError: If `inputs` holds something other than strings.
        ValueError: If `inputs` holds missing values and `allow_na` is not set.
    """
    if not isinstance(inputs, pd.Series):
        inputs = pd.Series(inputs, dtype="object")
    if inputs.empty:
        raise ValueError("inputs is empty")
    inferred = pd.api.types.infer_dtype(inputs, skipna=True)
    if inferred not in ("string", "empty"):
        raise TypeError(f"inputs must contain strings, got {inferred}")
    n_missing = int(inputs.isna().sum())
    if n_missing and not allow_na:
        raise ValueError(f"inputs contains {n_missing} missing value(s); drop them or pass allow_na=True")
    return inputs.astype("string") if allow_na else inputs


class WordAnnotator:
    """Counts occurrences of a list of words in text.

    Attributes:
        words: The words this annotator counts, in output column order.
        case: Whether matching is case-sensitive.
    """

    words: list[str]
    case: bool

    @staticmethod
    def _validate_words(words: list[str], *, case: bool) -> None:
        """Check that `words` is a non-empty list of distinct, non-empty strings.

        Args:
            words: The words to check.
            case: Whether matching is case-sensitive. When it is not, words differing only by
                case are duplicates, since they would produce identical columns.

        Raises:
            ValueError: If `words` is empty, contains an empty string, or contains duplicates.
        """
        if not words:
            raise ValueError("No words passed")
        if "" in words:
            raise ValueError("Empty string '' passed in word list")
        keys = words if case else [word.casefold() for word in words]
        duplicates = sorted({key for key, count in Counter(keys).items() if count > 1})
        if duplicates:
            raise ValueError(f"Duplicate words passed in list: {duplicates}")

    def __init__(self, words: Iterable[str] | None = None, *, case: bool = False) -> None:
        """Validate the words to count.

        Args:
            words: The words to count. Defaults to `DEFAULT_WORDS`.
            case: Whether matching is case-sensitive. Defaults to False, so the lowercase
                `DEFAULT_WORDS` match text of any case.

        Raises:
            TypeError: If `words` is a single word rather than a list of words.
            ValueError: If `words` is empty, contains an empty string, or contains duplicates.
        """
        if isinstance(words, str):
            hint = "pass None for default word list" if words == ALL else f"pass ['{words}'] instead"
            raise TypeError(f"words must be a list of strings or None, not a single word; {hint}")
        words = list(DEFAULT_WORDS if words is None else words)
        self._validate_words(words, case=case)
        self.words = words
        self.case = case

    @staticmethod
    def _annotate_word(inputs: pd.Series, word: str, *, method: WordMethod, case: bool) -> pd.Series:
        """Annotate inputs for a single word.

        Args:
            inputs: The texts to annotate.
            word: The word to match.
            method: Whether to count occurrences or flag containment.
            case: Whether matching is case-sensitive.

        Returns:
            Count, named after the word.

        Note:
            A word starting or ending with a non-word character (for example "'em") never
            matches, because the word boundary on that side cannot be satisfied.
        """
        pattern = rf"\b{re.escape(word)}\b"
        flags = 0 if case else re.IGNORECASE
        if method == "count":
            return inputs.str.count(pattern, flags=flags).rename(f"word_count__{word}")
        return inputs.str.contains(pattern, flags=flags).rename(f"word_contains__{word}")

    def annotate(
        self, inputs: list[str] | pd.Series, *, method: WordMethod = "count", allow_na: bool = False
    ) -> pd.DataFrame:
        """Annotate inputs for every word in `self.words`.

        Args:
            inputs: The texts to annotate.
            method: Whether to count occurrences or flag containment.
            allow_na: Whether missing values are allowed. When they are, a missing input yields
                `pd.NA` rather than zero occurrences.

        Returns:
            One column per word, in `self.words` order, sharing the index of `inputs`.

        Raises:
            TypeError: If `inputs` holds something other than strings.
            ValueError: If `method` is not a supported method, or if `inputs` holds missing
                values and `allow_na` is not set.
        """
        if method not in EXISTING_WORD_METHODS:
            raise ValueError(f"method must be one of {EXISTING_WORD_METHODS}, got '{method}'")
        prepared = _prepare_inputs(inputs, allow_na=allow_na)
        annotated = [self._annotate_word(prepared, word=w, method=method, case=self.case) for w in self.words]
        return pd.concat(annotated, axis=1)


class AssertionAnnotator:
    """Scores text against one or more assertions using their SetFit classifiers.

    Attributes:
        assertions: The assertions this annotator scores, in output column order.
        models: The loaded models, keyed by assertion. Empty when `lazy` is set.
        device: The device inference runs on.
        lazy: Whether models are loaded per call rather than held for the annotator's lifetime.
    """

    assertions: list[Assertion]
    models: dict[Assertion, SetFitModel]
    device: torch.device
    lazy: bool

    def _validate_assertions(self) -> None:
        """Check that `self.assertions` is non-empty, duplicate-free, and published.

        Raises:
            ValueError: If no assertions were passed, an assertion appears twice, or an
                assertion has no published model.
        """
        if not self.assertions:
            raise ValueError("No assertions passed")
        valid_assertions: set[str] = set()
        invalid_assertions: set[str] = set()
        for assertion in self.assertions:
            if assertion in valid_assertions or assertion in invalid_assertions:
                raise ValueError(f"Duplicate assertion '{assertion}' passed in list: {self.assertions}")
            if assertion not in EXISTING_ASSERTIONS:
                invalid_assertions.add(assertion)
            else:
                valid_assertions.add(assertion)
        if invalid_assertions:
            raise ValueError(f"Invalid assertions passed: {list(invalid_assertions)}")

    def _load_model(self, assertion: Assertion) -> SetFitModel:
        """Load one assertion's model onto the CPU.
        Model is loaded onto device if available with `_on_device` when predicting.

        Args:
            assertion: The assertion whose model to load.

        Returns:
            The model, on the CPU.
        """
        return SetFitModel.from_pretrained(f"StanfordSCALE/assertion_{assertion}", device="cpu")

    def _load_models(self) -> None:
        """Load every model in `self.assertions` that is not already loaded."""
        for assertion in tqdm(self.assertions):
            if assertion in self.models:
                continue
            self.models[assertion] = self._load_model(assertion)

    def __init__(
        self,
        assertions: Iterable[Assertion] | None = None,
        *,
        device: str | torch.device | None = None,
        lazy: bool = True,
    ) -> None:
        """Validate the requested assertions and, unless `lazy` is set, load their models.

        Args:
            assertions: The assertions to score. Defaults to `EXISTING_ASSERTIONS`, every
                published assertion, each of which downloads its own model when it is scored.
            device: The device to run inference on. Defaults to the best available
                accelerator.
            lazy: Whether to load each model only for as long as it is being used, instead of
                holding every model for the annotator's lifetime.

        Raises:
            TypeError: If `assertions` is a single assertion rather than a list of assertions.
            ValueError: If `assertions` is empty, contains duplicates, or names an assertion
                with no published model.
        """
        if isinstance(assertions, str):
            hint = "pass None for all existing assertions" if assertions == ALL else f"pass ['{assertions}'] instead"
            raise TypeError(f"assertions must be a list of assertions or None, not a single assertion; {hint}")
        self.assertions = list(EXISTING_ASSERTIONS if assertions is None else assertions)
        self._validate_assertions()
        self.device = torch.device(device) if device is not None else torch.device(get_device_name())
        self.lazy = lazy
        self.models = {}
        if not self.lazy:
            self._load_models()

    @contextmanager
    def _on_device(self, assertion: Assertion) -> Generator[SetFitModel]:
        """Make an assertion's model resident on `self.device` for the duration of the block.

        On exit the model is released if `self.lazy` is set, and otherwise moved back to the
        CPU, so at most one model occupies the accelerator at a time.

        Args:
            assertion: The assertion whose model to make resident.

        Yields:
            The model, on `self.device`.
        """
        model = self.models[assertion] if assertion in self.models else self._load_model(assertion)
        model.to(self.device)
        try:
            yield model
        finally:
            if self.lazy:
                # drop the last reference so the weights are freed; the caching allocator
                # reuses the blocks for the next model, so no empty_cache() is needed
                self.models.pop(assertion, None)
            else:
                self.models[assertion] = model.to("cpu")

    def _predict_single(
        self, inputs: list[str], assertion: Assertion, *, batch_size: int = 32, show_progress_bar: bool | None = None
    ) -> pd.Series:
        """Score inputs against a single assertion.

        Args:
            inputs: The texts to score.
            assertion: The assertion to score against.
            batch_size: The batch size to encode with.
            show_progress_bar: Whether to show a progress bar while encoding.

        Returns:
            The positive-class probability per input, named after the assertion.
        """
        with self._on_device(assertion) as model:
            output = model.predict_proba(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar)
        output = output[:, 1].tolist()
        return pd.Series(output, name=assertion)

    def predict_proba(
        self, inputs: list[str] | pd.Series, *, batch_size: int = 32, show_progress_bar: bool | None = None
    ) -> pd.DataFrame:
        """Score inputs against every assertion.

        Args:
            inputs: The texts to score.
            batch_size: The batch size to encode with.
            show_progress_bar: Whether to show a progress bar while encoding.

        Returns:
            The positive-class probability per input, with one column per assertion.

        Raises:
            TypeError: If `inputs` holds something other than strings.
            ValueError: If `inputs` is empty or holds missing values.
        """
        prepared = _prepare_inputs(inputs)
        texts = prepared.to_list()
        preds = {
            a: self._predict_single(texts, assertion=a, batch_size=batch_size, show_progress_bar=show_progress_bar)
            for a in self.assertions
        }
        return pd.DataFrame(preds).add_prefix("assertion__").set_axis(prepared.index)

    def predict(
        self,
        inputs: list[str] | pd.Series,
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Predict whether each input fires each assertion.

        Args:
            inputs: The texts to score.
            batch_size: The batch size to encode with.
            show_progress_bar: Whether to show a progress bar while encoding.
            threshold: The probability at or above which an assertion fires.

        Returns:
            Whether each input fires each assertion, with one column per assertion, sharing the
            index of `inputs`.

        Raises:
            TypeError: If `inputs` holds something other than strings.
            ValueError: If `inputs` is empty or holds missing values.
        """
        return self.predict_proba(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar) >= threshold

    def annotate(
        self,
        inputs: list[str] | pd.Series,
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Annotate inputs for every assertion, the way `WordAnnotator.annotate` does.

        Args:
            inputs: The texts to score.
            batch_size: The batch size to encode with.
            show_progress_bar: Whether to show a progress bar while encoding.
            threshold: The probability at or above which an assertion fires.

        Returns:
            Whether each input fires each assertion, with one column per assertion, sharing the
            index of `inputs`.

        Raises:
            TypeError: If `inputs` holds something other than strings.
            ValueError: If `inputs` is empty or holds missing values.
        """
        return self.predict(inputs, batch_size=batch_size, show_progress_bar=show_progress_bar, threshold=threshold)
