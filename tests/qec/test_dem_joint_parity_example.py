"""Check the fixed joint parity example against a rational reference table."""

import json
from fractions import Fraction

import pytest

from examples.qec.dem_joint_parity import main

pytestmark = pytest.mark.unit


def test_joint_parity_probabilities(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    probabilities = json.loads(capsys.readouterr().out, parse_float=Fraction)

    # No faults, both faults, the detector-only fault, and the shared fault.
    assert probabilities == {
        "00": Fraction(21, 32),
        "01": Fraction(1, 32),
        "10": Fraction(7, 32),
        "11": Fraction(3, 32),
    }
    assert sum(probabilities.values()) == 1
    assert probabilities["10"] + probabilities["11"] == Fraction(5, 16)
    assert probabilities["01"] + probabilities["11"] == Fraction(1, 8)
