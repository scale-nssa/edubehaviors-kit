# EduBehaviors-Kit

![PyPI version](https://img.shields.io/pypi/v/edubehaviors-kit.svg)

Python code for predicting and training with the EduBehaviors framework

* GitHub: https://github.com/scale-nssa/edubehaviors-kit/
* PyPI package: https://pypi.org/project/edubehaviors-kit/
* Created by: **[Xander Beberman](https://github.com/xanderbeberman), [Julian Bernado](https://github.com/julian-bernado), and [Ana T. Ribeiro](https://github.com/anatrindaderibeiro) at [The SCALE Initiative at Stanford University](https://scale.stanford.edu/)**.
* Free software: MIT License

## Features

* `WordAnnotator` — counts or flags a word list in text
* `AssertionAnnotator` — scores text against the EduBehaviors assertions with their SetFit models
* `standard_classifier` — ridge logistic regression with sensible defaults
* `ClassificationPipeline` — annotate, split, train and evaluate a labelled dataframe in one step

## Assertions

See the [assertions page](https://scale-nssa.github.io/edubehaviors-kit/assertions/) for the published
assertion classifiers with their agreement and test F1 scores.

## Usage

### Quickstart

```python
import pandas as pd

from edubehaviors import ClassificationPipeline

data = pd.read_csv("examples/talkmoves_tutor.csv")
pipeline = ClassificationPipeline(
    data,
    words="all",
    assertions=["sentence_has_a_question"],
    label_column="label_press_for_reasoning",
    group_column="transcript",
    random_state=2026_09_17,
)

print(pipeline.report())
```

See the [usage docs](https://scale-nssa.github.io/edubehaviors-kit/usage/) for details.

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://scale-nssa.github.io/edubehaviors-kit/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/edubehaviors-kit.git
cd edubehaviors-kit

# Install the project and its dev dependencies
uv sync
```

This installs the package in editable mode, so any changes you make to the source code are picked up immediately.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

EduBehaviors-Kit was created in 2026 by Xander Beberman, Julian Bernado, and Ana T. Ribeiro at The SCALE Initiative at Stanford University.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
