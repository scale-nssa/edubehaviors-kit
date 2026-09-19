"""Smoke tests for the annotators."""

import pandas as pd
import pytest

from edubehaviors import DEFAULT_WORDS, EXISTING_ASSERTIONS, AssertionAnnotator, WordAnnotator


class TestWordAnnotator:
    def test_defaults_to_the_default_words(self):
        assert WordAnnotator().words == list(DEFAULT_WORDS)

    def test_counts_occurrences(self, texts: list[str]):
        annotated = WordAnnotator(["you", "down"]).annotate(texts)
        assert list(annotated.columns) == ["word_count__you", "word_count__down"]
        assert len(annotated) == len(texts)
        assert annotated["word_count__you"].tolist() == [1, 0, 1, 0]
        assert annotated["word_count__down"].tolist() == [0, 1, 0, 0]

    def test_flags_containment(self, texts: list[str]):
        annotated = WordAnnotator(["you"]).annotate(texts, method="contains")
        assert list(annotated.columns) == ["word_contains__you"]
        assert annotated["word_contains__you"].tolist() == [True, False, True, False]

    def test_matching_is_case_insensitive_by_default(self):
        assert WordAnnotator(["why"]).annotate(["Why not?"])["word_count__why"].tolist() == [1]
        assert WordAnnotator(["why"], case=True).annotate(["Why not?"])["word_count__why"].tolist() == [0]

    def test_keeps_the_index_of_the_input(self):
        inputs = pd.Series(["you", "me"], index=[7, 9])
        assert WordAnnotator(["you"]).annotate(inputs).index.tolist() == [7, 9]

    def test_missing_values_need_opting_in(self):
        inputs = pd.Series(["you", None])
        with pytest.raises(ValueError, match="missing value"):
            WordAnnotator(["you"]).annotate(inputs)
        annotated = WordAnnotator(["you"]).annotate(inputs, allow_na=True)
        assert annotated["word_count__you"].tolist()[0] == 1
        assert annotated["word_count__you"].isna().tolist() == [False, True]

    @pytest.mark.parametrize(
        ("words", "match"),
        [([], "No words"), ([""], "Empty string"), (["a", "A"], "Duplicate")],
    )
    def test_rejects_unusable_word_lists(self, words: list[str], match: str):
        with pytest.raises(ValueError, match=match):
            WordAnnotator(words)

    def test_rejects_a_single_word(self):
        with pytest.raises(TypeError, match="not a single word"):
            WordAnnotator("you")

    def test_rejects_an_unknown_method(self, texts: list[str]):
        with pytest.raises(ValueError, match="method must be one of"):
            WordAnnotator(["you"]).annotate(texts, method="sometimes")  # ty: ignore[invalid-argument-type]

    def test_rejects_non_string_inputs(self):
        with pytest.raises(TypeError, match="must contain strings"):
            WordAnnotator(["you"]).annotate([1, 2])  # ty: ignore[invalid-argument-type]

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="empty"):
            WordAnnotator(["you"]).annotate([])


class TestAssertionAnnotator:
    """Construction and validation are offline; scoring runs against `stub_models`."""

    def test_defaults_to_every_published_assertion(self):
        assert AssertionAnnotator().assertions == list(EXISTING_ASSERTIONS)

    def test_lazy_construction_loads_nothing(self):
        assert AssertionAnnotator([EXISTING_ASSERTIONS[0]]).models == {}

    def test_rejects_unusable_assertion_lists(self):
        with pytest.raises(ValueError, match="No assertions"):
            AssertionAnnotator([])
        with pytest.raises(ValueError, match="Duplicate"):
            AssertionAnnotator([EXISTING_ASSERTIONS[0], EXISTING_ASSERTIONS[0]])
        with pytest.raises(ValueError, match="Invalid assertions"):
            AssertionAnnotator(["sentence_reads_your_mind"])  # ty: ignore[invalid-argument-type]

    def test_rejects_a_single_assertion(self):
        with pytest.raises(TypeError, match="not a single assertion"):
            AssertionAnnotator(EXISTING_ASSERTIONS[0])  # ty: ignore[invalid-argument-type]

    def test_predict_proba_returns_a_column_per_assertion(self, stub_models: None, texts: list[str]):
        assertions = list(EXISTING_ASSERTIONS[:2])
        scored = AssertionAnnotator(assertions).predict_proba(texts)
        assert list(scored.columns) == [f"assertion__{a}" for a in assertions]
        assert len(scored) == len(texts)
        assert scored.to_numpy().min() >= 0.0
        assert scored.to_numpy().max() <= 1.0

    def test_predict_thresholds_the_probabilities(self, stub_models: None, texts: list[str]):
        annotator = AssertionAnnotator([EXISTING_ASSERTIONS[0]])
        column = f"assertion__{EXISTING_ASSERTIONS[0]}"
        assert annotator.predict(texts)[column].tolist() == [True, False, True, False]
        assert annotator.predict(texts, threshold=0.95)[column].tolist() == [False] * len(texts)

    def test_annotate_matches_predict(self, stub_models: None, texts: list[str]):
        annotator = AssertionAnnotator([EXISTING_ASSERTIONS[0]])
        pd.testing.assert_frame_equal(annotator.annotate(texts), annotator.predict(texts))

    def test_lazy_annotator_releases_models_between_calls(self, stub_models: None, texts: list[str]):
        annotator = AssertionAnnotator([EXISTING_ASSERTIONS[0]], lazy=True)
        annotator.predict_proba(texts)
        assert annotator.models == {}

    def test_eager_annotator_retains_models_on_the_cpu(self, stub_models: None, texts: list[str]):
        annotator = AssertionAnnotator([EXISTING_ASSERTIONS[0]], lazy=False)
        assert set(annotator.models) == {EXISTING_ASSERTIONS[0]}
        annotator.predict_proba(texts)
        assert annotator.models[EXISTING_ASSERTIONS[0]].device == "cpu"

    def test_keeps_the_index_of_the_input(self, stub_models: None):
        inputs = pd.Series(["Why?", "Sit down."], index=[4, 6])
        scored = AssertionAnnotator([EXISTING_ASSERTIONS[0]]).predict_proba(inputs)
        assert scored.index.tolist() == [4, 6]
