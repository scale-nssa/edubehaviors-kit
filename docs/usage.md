# Usage

## Quickstart

`ClassificationPipeline` takes a dataframe with a text column and a label column, then annotates the
text, splits the rows, fits a classifier and scores the held-out rows — all while it is being
built. The result is ready to read straight away:

```python
import pandas as pd

from edubehaviors import ClassificationPipeline

data = pd.read_csv("examples/talkmoves_tutor.csv")

pipeline = ClassificationPipeline(
    data,
    words="all",  # every word in DEFAULT_WORDS
    assertions=["sentence_has_a_question", "sentence_has_explanation_or_reasoning"],
    label_column="label_press_for_reasoning",
    random_state=2026_09_17,
)

print(pipeline.metrics)
print(pipeline.report())
```

Labels are used as they appear, so a boolean column gives a binary prediction problem and a
column of category names gives a multiclass one.

## Choosing features

Each feature family takes a list, the string `"all"`, or `None` to skip it. At least one family
must be requested:

```python
ClassificationPipeline(data, words="all")  # every default word, no assertions
ClassificationPipeline(data, words=["why", "how", "because"])  # just these words
ClassificationPipeline(data, assertions=["sentence_has_a_question"])  # assertions only
ClassificationPipeline(data, words="all", assertions="all")  # everything
```

Every assertion downloads its own SetFit model, so `assertions="all"` pulls one model per entry
in `EXISTING_ASSERTIONS`. This can get large and slow quickly, so try starting with the handful
you care most about.

Assertion features are positive-class probabilities by default; pass
`assertion_method="binary"` for boolean. Pass `lazy=False` to keep the
weights loaded if you plan to call `predict` repeatedly afterwards.

## Group-aware splitting

If utterances must be kept in groups during splitting, pass a `group_column` and the
pipeline will do grouped and stratified splitting:

```python
pipeline = ClassificationPipeline(
    data,
    words="all",
    label_column="label_press_for_reasoning",
    group_column="transcript",
    random_state=0,
)
pipeline.split_strategy  # 'stratified_grouped'
```

Note that with `group_column` set, `test_size` is approximate since whole groups move together,
so the closest groups available to your `group_size` is used.

## Reading the result

```python
pipeline.predicted  # held-out rows with their text, true label, prediction and whether they match
pipeline.predictions  # held-out model predictions only
pipeline.annotated  # the original data plus features plus the split column
pipeline.metrics  # held-out accuracy and macro precision / recall / F1
pipeline.features  # the feature matrix, one row per input row
pipeline.split  # 'train' / 'test' per row
pipeline.coefficients  # fitted coefficients, indexed by feature name
pipeline.classifier  # the fitted scikit-learn estimator
pipeline.predict(["why do you think that works?"])  # label new text
```


## Annotating without a classifier

The annotators work on their own, on a list of strings or a Series, and both preserve the index
of their input. Each takes a list of words or assertions, or None for the full default set:

```python
from edubehaviors import AssertionAnnotator, WordAnnotator

word_features = WordAnnotator().annotate(data.sentence)  # DEFAULT_WORDS
annotator = AssertionAnnotator(["sentence_has_a_question"])
assertion_flags = annotator.annotate(data.sentence)  # same as predict()
assertion_scores = annotator.predict_proba(data.sentence)  # probabilities instead
```
