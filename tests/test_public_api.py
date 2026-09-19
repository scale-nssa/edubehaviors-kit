"""Smoke tests for what an installed copy of the package exposes."""

import importlib.metadata
import importlib.resources
from typing import get_args

import pytest

import edubehaviors
from edubehaviors import DEFAULT_WORDS, EXISTING_ASSERTIONS, Assertion

DISTRIBUTION = "edubehaviors-kit"


def test_import():
    """Verify the package can be imported."""
    assert edubehaviors


@pytest.mark.parametrize("name", edubehaviors.__all__)
def test_exported_name_is_reachable(name: str):
    """Every name in `__all__` resolves, so `from edubehaviors import ...` works."""
    assert getattr(edubehaviors, name, None) is not None


def test_declared_console_scripts_are_importable():
    """Every declared entry point loads.

    A script pointing at a module that does not exist installs cleanly and only fails when
    the user runs it, so the packaging metadata is checked here rather than by hand.
    """
    distribution = importlib.metadata.distribution(DISTRIBUTION)
    scripts = [ep for ep in distribution.entry_points if ep.group == "console_scripts"]
    for entry_point in scripts:
        entry_point.load()


def test_py_typed_is_packaged():
    """The `Typing :: Typed` classifier is only honest if the marker ships."""
    assert importlib.resources.files("edubehaviors").joinpath("py.typed").is_file()


def test_existing_assertions_match_the_type():
    """`EXISTING_ASSERTIONS` stays in step with the `Assertion` literal it is derived from."""
    assert EXISTING_ASSERTIONS
    assert EXISTING_ASSERTIONS == get_args(Assertion)
    assert len(set(EXISTING_ASSERTIONS)) == len(EXISTING_ASSERTIONS)


def test_default_words_are_usable():
    """`DEFAULT_WORDS` is distinct and lowercase, as case-insensitive matching assumes."""
    assert DEFAULT_WORDS
    assert len(set(DEFAULT_WORDS)) == len(DEFAULT_WORDS)
    assert all(word == word.casefold() for word in DEFAULT_WORDS)
    assert "" not in DEFAULT_WORDS
