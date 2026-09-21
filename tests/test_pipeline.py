"""Smoke tests for the end-to-end pipeline.

Every test here configures word features only, so the pipeline runs without downloading a
single assertion model.
"""

import numpy as np
import pandas as pd
import pytest

from edubehaviors import ClassificationPipeline
from edubehaviors.pipeline import SPLIT_COLUMN

TEST_WORDS = ["why", "think", "answer", "down", "now", "your"]
"""A small word list that separates the two labels in `labelled_data`."""


@pytest.fixture
def pipeline(labelled_data: pd.DataFrame) -> ClassificationPipeline:
    return ClassificationPipeline(labelled_data, words=TEST_WORDS, random_state=0)


class TestFitting:
    def test_builds_features_split_and_metrics(self, pipeline: ClassificationPipeline, labelled_data: pd.DataFrame):
        assert list(pipeline.features.columns) == [f"word_count__{w}" for w in TEST_WORDS]
        assert pipeline.features.index.equals(labelled_data.index)
        assert set(pipeline.split.unique()) == {"train", "test"}
        assert pipeline.metrics["n_train"] + pipeline.metrics["n_test"] == len(labelled_data)

    def test_separable_labels_score_well(self, pipeline: ClassificationPipeline):
        assert pipeline.metrics["f1_macro"] > 0.9

    def test_metrics_report_the_documented_keys(self, pipeline: ClassificationPipeline):
        assert set(pipeline.metrics) == {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "n_train",
            "n_test",
        }

    def test_test_size_is_respected(self, labelled_data: pd.DataFrame):
        pipeline = ClassificationPipeline(labelled_data, words=TEST_WORDS, test_size=0.5, random_state=0)
        assert pipeline.metrics["n_test"] == pytest.approx(len(labelled_data) / 2, abs=1)

    def test_is_reproducible_given_a_seed(self, labelled_data: pd.DataFrame):
        first = ClassificationPipeline(labelled_data, words=TEST_WORDS, random_state=7)
        second = ClassificationPipeline(labelled_data, words=TEST_WORDS, random_state=7)
        pd.testing.assert_series_equal(first.split, second.split)
        assert first.metrics == second.metrics

    def test_words_all_uses_the_default_list(self, labelled_data: pd.DataFrame):
        pipeline = ClassificationPipeline(labelled_data, words="all", random_state=0)
        assert pipeline.word_annotator is not None
        assert pipeline.assertion_annotator is None
        assert pipeline.features.shape[1] == len(pipeline.word_annotator.words)

    def test_prebuilt_features_are_reused(self, pipeline: ClassificationPipeline, labelled_data: pd.DataFrame):
        reused = ClassificationPipeline(labelled_data, words=TEST_WORDS, random_state=1, features=pipeline.features)
        pd.testing.assert_frame_equal(reused.features, pipeline.features)


class TestSplitting:
    def test_stratified_by_default(self, pipeline: ClassificationPipeline):
        assert pipeline.split_strategy == "stratified"

    def test_unstratified_split_is_random(self, labelled_data: pd.DataFrame):
        pipeline = ClassificationPipeline(labelled_data, words=TEST_WORDS, stratify=False, random_state=0)
        assert pipeline.split_strategy == "random"

    @pytest.mark.parametrize(("stratify", "strategy"), [(True, "stratified_grouped"), (False, "grouped")])
    def test_groups_do_not_straddle_the_split(self, labelled_data: pd.DataFrame, stratify: bool, strategy: str):
        pipeline = ClassificationPipeline(
            labelled_data,
            words=TEST_WORDS,
            group_column="transcript",
            stratify=stratify,
            random_state=0,
        )
        assert pipeline.split_strategy == strategy
        per_group = labelled_data["transcript"].groupby(pipeline.split).apply(set)
        assert not per_group["train"] & per_group["test"]

    def test_too_few_groups_is_an_error(self, labelled_data: pd.DataFrame):
        two_groups = labelled_data.assign(transcript=["a", "b"] * (len(labelled_data) // 2))
        with pytest.raises(ValueError, match="needs at least"):
            ClassificationPipeline(two_groups, words=TEST_WORDS, group_column="transcript", test_size=0.1)


class TestOutputs:
    def test_report_is_a_readable_string(self, pipeline: ClassificationPipeline):
        report = pipeline.report()
        assert "precision" in report
        assert "question" in report

    def test_evaluate_scores_either_split(self, pipeline: ClassificationPipeline):
        assert pipeline.evaluate(split="train")["n_train"] == pipeline.metrics["n_train"]
        assert pipeline.evaluate(split="test") == pipeline.metrics

    def test_evaluate_rejects_an_unknown_split(self, pipeline: ClassificationPipeline):
        with pytest.raises(ValueError, match="split must be one of"):
            pipeline.evaluate(split="holdout")  # ty: ignore[invalid-argument-type]

    def test_predicts_unseen_text(self, pipeline: ClassificationPipeline):
        predicted = pipeline.predict(["Why do you think that is the answer?", "Put your pencils down now."])
        assert predicted.tolist() == ["question", "directive"]

    def test_predict_proba_gives_a_column_per_label(self, pipeline: ClassificationPipeline):
        probabilities = pipeline.predict_proba(["Why do you think?"])
        assert list(probabilities.columns) == ["directive", "question"]
        assert probabilities.to_numpy().sum() == pytest.approx(1.0)

    def test_predicted_pairs_truth_with_prediction(self, pipeline: ClassificationPipeline):
        predicted = pipeline.predicted
        assert list(predicted.columns) == ["sentence", "label", "prediction", "correct"]
        assert len(predicted) == pipeline.metrics["n_test"]
        assert predicted["correct"].mean() == pytest.approx(pipeline.metrics["accuracy"])

    def test_binary_coefficients_are_a_series(self, pipeline: ClassificationPipeline):
        coefficients = pipeline.coefficients
        assert isinstance(coefficients, pd.Series)
        assert coefficients.index.tolist() == list(pipeline.features.columns)

    def test_multiclass_coefficients_are_a_frame(self, labelled_data: pd.DataFrame):
        labels = labelled_data["label"].to_list()
        labels[::4] = ["other"] * len(labels[::4])
        pipeline = ClassificationPipeline(labelled_data.assign(label=labels), words=TEST_WORDS, random_state=0)
        coefficients = pipeline.coefficients
        assert isinstance(coefficients, pd.DataFrame)
        assert coefficients.shape == (len(TEST_WORDS), 3)

    def test_annotated_joins_data_features_and_split(
        self, pipeline: ClassificationPipeline, labelled_data: pd.DataFrame
    ):
        annotated = pipeline.annotated
        assert SPLIT_COLUMN in annotated.columns
        assert set(labelled_data.columns) <= set(annotated.columns)
        assert set(pipeline.features.columns) <= set(annotated.columns)


class TestValidation:
    def test_requires_at_least_one_feature_family(self, labelled_data: pd.DataFrame):
        with pytest.raises(ValueError, match="No features requested"):
            ClassificationPipeline(labelled_data)

    def test_rejects_a_single_word(self, labelled_data: pd.DataFrame):
        with pytest.raises(TypeError, match="not a single value"):
            ClassificationPipeline(labelled_data, words="why")

    def test_rejects_a_single_assertion(self, labelled_data: pd.DataFrame):
        with pytest.raises(TypeError, match="not a single value"):
            ClassificationPipeline(labelled_data, assertions="sentence_has_a_question")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("test_size", [0.0, 1.0, 1.5])
    def test_rejects_a_test_size_outside_the_unit_interval(self, labelled_data: pd.DataFrame, test_size: float):
        with pytest.raises(ValueError, match="test_size must be"):
            ClassificationPipeline(labelled_data, words=TEST_WORDS, test_size=test_size)

    def test_rejects_an_unknown_assertion_method(self, labelled_data: pd.DataFrame):
        with pytest.raises(ValueError, match="assertion_method must be one of"):
            ClassificationPipeline(labelled_data, words=TEST_WORDS, assertion_method="logits")  # ty: ignore[invalid-argument-type]

    def test_rejects_a_duplicated_index(self, labelled_data: pd.DataFrame):
        duplicated = pd.concat([labelled_data, labelled_data])
        with pytest.raises(ValueError, match="duplicated index"):
            ClassificationPipeline(duplicated, words=TEST_WORDS)

    def test_rejects_a_single_label(self, labelled_data: pd.DataFrame):
        with pytest.raises(ValueError, match="fewer than 2 distinct labels"):
            ClassificationPipeline(labelled_data.assign(label="question"), words=TEST_WORDS)

    @pytest.mark.parametrize("missing", [np.nan, None, pd.NA])
    def test_rejects_missing_labels(self, labelled_data: pd.DataFrame, missing: object):
        labels = labelled_data["label"].astype("object")
        labels.iloc[:3] = missing
        with pytest.raises(ValueError, match="contains missing values"):
            ClassificationPipeline(labelled_data.assign(label=labels), words=TEST_WORDS)

    def test_rejects_missing_labels_before_annotating(
        self, labelled_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ):
        """Annotation is the expensive step, so missing labels must fail ahead of it."""

        def fail(*args: object, **kwargs: object) -> pd.DataFrame:
            raise AssertionError("annotate() ran despite missing labels")

        monkeypatch.setattr(ClassificationPipeline, "annotate", fail)
        labels = labelled_data["label"].astype("object")
        labels.iloc[:3] = np.nan
        with pytest.raises(ValueError, match="contains missing values"):
            ClassificationPipeline(labelled_data.assign(label=labels), words=TEST_WORDS)

    def test_missing_labels_are_reported_before_too_few_labels(self, labelled_data: pd.DataFrame):
        """nunique() skips NaN, so a sparse single-label column must not be blamed on the count."""
        labels = pd.Series(["question"] * len(labelled_data), dtype="object")
        labels.iloc[:3] = np.nan
        with pytest.raises(ValueError, match="contains missing values"):
            ClassificationPipeline(labelled_data.assign(label=labels), words=TEST_WORDS)

    def test_rejects_misaligned_prebuilt_features(self, pipeline: ClassificationPipeline, labelled_data: pd.DataFrame):
        shifted = pipeline.features.set_axis(labelled_data.index + 1)
        with pytest.raises(ValueError, match="must share the index"):
            ClassificationPipeline(labelled_data, words=TEST_WORDS, features=shifted)

    def test_missing_column_is_an_error(self, labelled_data: pd.DataFrame):
        with pytest.raises(KeyError):
            ClassificationPipeline(labelled_data, words=TEST_WORDS, label_column="absent")
