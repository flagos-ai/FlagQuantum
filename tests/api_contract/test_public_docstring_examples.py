"""Public workflow examples remain executable and free of external effects."""

import doctest

import pytest

import flagquantum as fq

pytestmark = pytest.mark.unit


def test_primary_public_docstring_examples() -> None:
    runner = doctest.DocTestRunner()
    finder = doctest.DocTestFinder()
    entries = (fq.Circuit, fq.Module, fq.compile, fq.plan, fq.run, fq.train)

    for entry in entries:
        name = f"{entry.__module__}.{entry.__name__}"
        for example in finder.find(entry, name=name):
            runner.run(example)

    failures, _ = runner.summarize()
    assert failures == 0
