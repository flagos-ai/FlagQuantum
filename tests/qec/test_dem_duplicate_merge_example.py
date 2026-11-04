"""Check the duplicate merge example against exact rational probabilities."""

import json
from fractions import Fraction

import pytest

from examples.qec.dem_duplicate_merge import main

pytestmark = pytest.mark.unit


def test_duplicate_merge_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    report = json.loads(capsys.readouterr().out, parse_float=Fraction)
    assert report == {
        "independent_count": 1,
        "independent_probability": Fraction(5, 16),
        "exclusive_detector_rate": Fraction(3, 8),
        "exclusive_merge": "rejected",
    }
