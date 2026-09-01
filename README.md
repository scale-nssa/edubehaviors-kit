# EduBehaviors-Kit

![PyPI version](https://img.shields.io/pypi/v/edubehaviors-kit.svg)

Python code for predicting and training with the EduBehaviors framework

* GitHub: https://github.com/xanderbeberman/edubehaviors-kit/
* PyPI package: https://pypi.org/project/edubehaviors-kit/
* Created by: **[Xander Beberman](https://scale.stanford.edu/)** | GitHub https://github.com/xanderbeberman | PyPI https://pypi.org/user/xanderbeberman/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://xanderbeberman.github.io/edubehaviors/
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

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `edubehaviors`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

EduBehaviors-Kit was created in 2026 by Xander Beberman.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
