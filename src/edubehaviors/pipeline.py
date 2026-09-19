"""End-to-end pipeline from a labelled dataframe to a fitted classifier."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, get_args
from warnings import warn

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)

from .annotation import AssertionAnnotator, WordAnnotator, WordMethod
from .constants import ALL, DEFAULT_WORDS, EXISTING_ASSERTIONS, Assertion
from .models import standard_classifier

Split = Literal["train", "test"]
EXISTING_SPLITS: tuple[Split, ...] = get_args(Split)
"""The splits `ClassificationPipeline` assigns rows to."""

AssertionMethod = Literal["proba", "binary"]
EXISTING_ASSERTION_METHODS: tuple[AssertionMethod, ...] = get_args(AssertionMethod)
"""The assertion feature types `ClassificationPipeline` supports."""

SplitStrategy = Literal["random", "stratified", "grouped", "stratified_grouped"]
"""How a split was produced."""

SPLIT_COLUMN = "split"
"""The name of the split column `ClassificationPipeline` builds."""


class ClassificationPipeline:
    """Takes in a labeled DataFrame, annotates it, splits into train/test, trains a
    standard classifier, and scores the test set.

        pipeline = ClassificationPipeline(data, words="all", label_column="label_press_for_reasoning")
        print(pipeline.report())

    Word features are counted with `WordAnnotator` and assertion features are scored with
    `AssertionAnnotator`. To use all default words or existing annotators, pass `"all"`.
    Otherwise, pass a list of words or assertions, or `None` to skip an annotator. At
    least one of `words` or `assertions` must be passed.

    Labels are used exactly as they appear in `label_column`, so a boolean column gives a binary
    problem and a column of names gives a multiclass one, with no relabelling behind your back.

    Attributes:
        word_annotator: The word annotator, or None when word features are off.
        assertion_annotator: The assertion annotator, or None when assertion features are off.
        text_column: The column holding the text that was annotated.
        label_column: The column holding the labels being predicted.
        group_column: The column that was kept out of both splits at once, or None.
        word_method: Whether word features count occurrences or flag containment.
        assertion_method: Whether assertion features are probabilities or booleans.
        test_size: The fraction of the data held out for testing.
        stratify: Whether the split preserved the label distribution.
        random_state: The seed the split and the classifier used.
        batch_size: The batch size assertion scoring encoded with.
        show_progress_bar: Whether assertion scoring showed a progress bar.
        data: The frame the pipeline was built from.
        features: The feature matrix, sharing the index of `data`.
        outcome: The labels, as they appeared in `data`.
        split: Whether each row is in the train or test split.
        split_strategy: How the split was produced.
        groups: The group of each row, or None when `group_column` is not set.
        classifier: The classifier, fitted on the train split.
        metrics: How the classifier scored on the test split.
    """

    word_annotator: WordAnnotator | None
    assertion_annotator: AssertionAnnotator | None
    text_column: str
    label_column: str
    group_column: str | None
    word_method: WordMethod
    assertion_method: AssertionMethod
    test_size: float
    stratify: bool
    random_state: int | RandomState | None
    batch_size: int
    show_progress_bar: bool | None

    data: pd.DataFrame
    features: pd.DataFrame
    outcome: pd.Series
    split: pd.Series
    split_strategy: SplitStrategy
    groups: pd.Series | None
    classifier: LogisticRegressionCV
    metrics: dict[str, float]
    _split_predictions: dict[Split, pd.Series]

    def __init__(
        self,
        data: pd.DataFrame,
        words: Iterable[str] | Literal["all"] | None = None,
        assertions: Iterable[Assertion] | Literal["all"] | None = None,
        *,
        text_column: str = "sentence",
        label_column: str = "label",
        group_column: str | None = None,
        word_method: WordMethod = "count",
        case: bool = False,
        assertion_method: AssertionMethod = "proba",
        lazy: bool = True,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int | RandomState | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        features: pd.DataFrame | None = None,
    ) -> None:
        """Annotate `data`, split it, fit the classifier, and score the held-out split.

        Args:
            data: The frame holding the text column, the label column, and the group column
                when one is given.
            words: The words to count, `"all"` for `DEFAULT_WORDS`, or None to skip word
                features. An empty iterable also skips them.
            assertions: The assertions to score, `"all"` for `EXISTING_ASSERTIONS`, or None to
                skip assertion features. An empty iterable also skips them. Every assertion
                scored downloads its own model, so `"all"` downloads one per published
                assertion.
            text_column: The column holding the text to annotate.
            label_column: The column holding the labels to predict. Its values are used as they
                are, so binarize beforehand if that is what you want.
            group_column: The column whose values must not straddle the split, for example a
                transcript id, so that rows from one transcript stay on one side.
            word_method: Whether word features count occurrences or flag containment.
            case: Whether word matching is case-sensitive.
            assertion_method: Whether assertion features are positive-class probabilities
                (`"proba"`) or booleans thresholded at 0.5 (`"binary"`).
            lazy: Whether assertion models are loaded per call rather than held for the
                pipeline's lifetime. Pass False when calling `predict` repeatedly afterwards.
            test_size: The fraction of the data to hold out. With `group_column` set this is
                approximate, since whole groups move together: the closest of
                `round(1 / test_size)` grouped folds is used, and uneven groups make some folds
                much larger than others.
            stratify: Whether the split preserves the label distribution. With `group_column`
                set this is best-effort, since whole groups move together.
            random_state: The seed the split and the classifier use.
            batch_size: The batch size assertion scoring encodes with.
            show_progress_bar: Whether assertion scoring shows a progress bar.
            features: A feature matrix from another pipeline's `features` or `annotate(data)`,
                to skip re-annotating when only the split or the seed changed.

        Raises:
            KeyError: If a configured column is missing from `data`.
            TypeError: If `words` or `assertions` is a single value, or the texts are not
                strings.
            ValueError: If neither words nor assertions were requested, `assertion_method` is
                not supported, `test_size` is not a fraction, `data` has a duplicated index or
                fewer than two distinct labels, `features` does not share the index of `data`,
                or the split leaves a single label in the train split.

        Warns:
            UserWarning: If the test split holds a single label, which makes its metrics
                degenerate.
        """
        requested_words: list[str] | None = None
        if isinstance(words, str):
            if words != ALL:
                raise TypeError(f"words must be a list, '{ALL}', or None, not a single value; pass ['{words}'] instead")
            requested_words = list(DEFAULT_WORDS)
        elif words is not None:
            requested_words = list(words) or None
        requested_assertions: list[Assertion] | None = None
        if isinstance(assertions, str):
            if assertions != ALL:
                raise TypeError(
                    f"assertions must be a list, '{ALL}', or None, not a single value; pass ['{assertions}'] instead"
                )
            requested_assertions = list(EXISTING_ASSERTIONS)
        elif assertions is not None:
            requested_assertions = list(assertions) or None
        if requested_words is None and requested_assertions is None:
            raise ValueError(f"No features requested; pass words, assertions, or both (either can be '{ALL}')")
        if assertion_method not in EXISTING_ASSERTION_METHODS:
            raise ValueError(f"assertion_method must be one of {EXISTING_ASSERTION_METHODS}, got '{assertion_method}'")
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")
        if data.index.has_duplicates:
            duplicates = data.index[data.index.duplicated()].unique().tolist()[:5]
            raise ValueError(f"data has a duplicated index, which misaligns features; duplicates include {duplicates}")
        self.word_annotator = None if requested_words is None else WordAnnotator(requested_words, case=case)
        self.assertion_annotator = (
            None if requested_assertions is None else AssertionAnnotator(requested_assertions, lazy=lazy)
        )
        self.text_column = text_column
        self.label_column = label_column
        self.group_column = group_column
        self.word_method = word_method
        self.assertion_method = assertion_method
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.data = data
        self.outcome = data[label_column]
        if self.outcome.nunique() < 2:
            raise ValueError(
                f"label column '{label_column}' has fewer than 2 distinct labels; at least 2 are needed to fit"
            )
        self.groups = None if group_column is None else data[group_column]
        if features is None:
            features = self.annotate(data)
        elif not features.index.equals(data.index):
            raise ValueError("features must share the index of data; pass a frame from this pipeline's annotate()")
        self.features = features
        self.split, self.split_strategy = self._make_split()
        is_train = self.split == "train"
        if self.outcome[is_train].nunique() < 2:
            raise ValueError("train split holds a single label; adjust test_size, the seed, or the grouping")
        if self.outcome[~is_train].nunique() < 2:
            warn("test split holds a single label, so its metrics are degenerate", UserWarning, stacklevel=2)
        self.classifier = standard_classifier(random_state=random_state).fit(
            features.loc[is_train], self.outcome[is_train]
        )
        self._split_predictions = {}
        self.metrics = self.evaluate()

    def annotate(self, data: pd.DataFrame | pd.Series | list[str]) -> pd.DataFrame:
        """Build the feature matrix for `data`.

        Args:
            data: A frame holding the text column, or the texts themselves.

        Returns:
            The word features followed by the assertion features, sharing the index of `data`.

        Raises:
            KeyError: If `data` is a frame without the text column.
            TypeError: If the texts are not strings.
            ValueError: If `data` is empty, or any text is missing.
        """
        text = data[self.text_column] if isinstance(data, pd.DataFrame) else data
        frames = []
        if self.word_annotator is not None:
            frames.append(self.word_annotator.annotate(text, method=self.word_method))
        if self.assertion_annotator is not None:
            score = (
                self.assertion_annotator.predict_proba
                if self.assertion_method == "proba"
                else self.assertion_annotator.predict
            )
            frames.append(score(text, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar))
        return pd.concat(frames, axis=1)

    def _make_split(self) -> tuple[pd.Series, SplitStrategy]:
        """Assign every row to the train or test split.

        Returns:
            The split of each row, and how it was produced.

        Raises:
            ValueError: If there are too few groups for `test_size`.
        """
        index = self.outcome.index
        if self.groups is None:
            train_positions, test_positions = train_test_split(
                np.arange(len(index)),
                test_size=self.test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=self.outcome if self.stratify else None,
            )
            strategy: SplitStrategy = "stratified" if self.stratify else "random"
        else:
            rows = np.zeros((len(index), 1))
            if self.stratify:
                # fold sizes vary as much as group sizes do
                n_splits = max(2, round(1 / self.test_size))
                n_groups = int(self.groups.nunique())
                if n_groups < n_splits:
                    raise ValueError(
                        f"test_size={self.test_size} needs at least {n_splits} groups in "
                        f"'{self.group_column}', got {n_groups}; raise test_size or pass stratify=False"
                    )
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                folds = list(splitter.split(rows, self.outcome, groups=self.groups))
                train_positions, test_positions = min(
                    folds, key=lambda fold: abs(len(fold[1]) / len(index) - self.test_size)
                )
                strategy = "stratified_grouped"
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
                train_positions, test_positions = next(splitter.split(rows, self.outcome, groups=self.groups))
                strategy = "grouped"
        split = pd.Series("train", index=index, name=SPLIT_COLUMN, dtype="object")
        split.iloc[test_positions] = "test"
        return split, strategy

    def _predict_split(self, split: Split) -> pd.Series:
        """Predict the labels of one split with the fitted classifier, once.

        Args:
            split: The split to predict.

        Returns:
            The predicted label of each row in that split.

        Raises:
            ValueError: If `split` is not a supported split.
        """
        if split not in EXISTING_SPLITS:
            raise ValueError(f"split must be one of {EXISTING_SPLITS}, got '{split}'")
        if split not in self._split_predictions:
            rows = self.features.loc[self.split == split]
            self._split_predictions[split] = pd.Series(
                self.classifier.predict(rows), index=rows.index, name="prediction"
            )
        return self._split_predictions[split]

    def evaluate(self, *, split: Split = "test") -> dict[str, float]:
        """Get macro-averaged metrics for specified split using fitted classifier.

        Args:
            split: The split to score.

        Returns:
            Accuracy, macro precision, macro recall and macro F1, with the size of each split.

        Raises:
            ValueError: If `split` is not a supported split.
        """
        predicted = self._predict_split(split)
        true = self.outcome[self.split == split]
        precision, recall, f1, _ = precision_recall_fscore_support(true, predicted, average="macro", zero_division=0)
        return {
            "accuracy": float(accuracy_score(true, predicted)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "n_train": int((self.split == "train").sum()),
            "n_test": int((self.split == "test").sum()),
        }

    def report(self, *, split: Split = "test", digits: int = 2) -> str:
        """Return classification report for specified split.

        Args:
            split: The split to report on.
            digits: The number of decimal places to report.

        Returns:
            SKlearn classification report

        Raises:
            ValueError: If `split` is not a supported split.
        """
        predicted = self._predict_split(split)
        true = self.outcome[self.split == split]
        return classification_report(true, predicted, digits=digits, zero_division=0)

    def predict(self, data: pd.DataFrame | pd.Series | list[str]) -> pd.Series:
        """Predict labels for text the pipeline has not seen.

        Args:
            data: A frame holding the text column, or the texts themselves.

        Returns:
            The predicted label per row, sharing the index of `data`.
        """
        features = self.annotate(data)[self.features.columns]
        return pd.Series(self.classifier.predict(features), index=features.index, name="prediction")

    def predict_proba(self, data: pd.DataFrame | pd.Series | list[str]) -> pd.DataFrame:
        """Predict label probabilities for text the pipeline has not seen.

        Args:
            data: A frame holding the text column, or the texts themselves.

        Returns:
            One column per label, sharing the index of `data`.
        """
        features = self.annotate(data)[self.features.columns]
        return pd.DataFrame(
            self.classifier.predict_proba(features), index=features.index, columns=self.classifier.classes_
        )

    @property
    def predictions(self) -> pd.Series:
        """The predicted label of each held-out row, which `metrics` is computed from."""
        return self._predict_split("test")

    @property
    def predicted(self) -> pd.DataFrame:
        """The held-out rows with their true and predicted labels."""
        is_test = self.split == "test"
        predictions = self.predictions
        y_true = self.outcome[is_test]
        return pd.DataFrame(
            {
                self.text_column: self.data.loc[is_test, self.text_column],
                self.label_column: y_true,
                "prediction": predictions,
                "correct": y_true == predictions,
            }
        )

    @property
    def annotated(self) -> pd.DataFrame:
        """The data with its features and split assignment, for inspection or export.

        Note:
            A `split` column already in the data is replaced by the pipeline's own.
        """
        data = self.data.drop(columns=[SPLIT_COLUMN], errors="ignore")
        return pd.concat([data, self.features, self.split], axis=1)

    @property
    def coefficients(self) -> pd.Series | pd.DataFrame:
        """The fitted coefficients, indexed by feature name. Binary problems give a
        Series; multiclass problems give one column per label.
        """
        coefficients = self.classifier.coef_
        if coefficients.shape[0] == 1:
            return pd.Series(coefficients[0], index=self.features.columns, name="coefficient")
        return pd.DataFrame(coefficients.T, index=self.features.columns, columns=self.classifier.classes_)
