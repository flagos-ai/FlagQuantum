"""The public numeric tutorial runs without physical-wire handling or decoding."""

from fractions import Fraction

import pytest

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.composition.numeric_program.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.composition.numeric_program.main import build_program

pytestmark = pytest.mark.integration


def test_numeric_tutorial_circuit_and_decoded_values() -> None:
    program = build_program()
    compare_states(program, reference_circuit(), ("x", "doubled", "result"))
    distribution = program.numeric_probabilities(
        fq.run(program), registers=("x", "doubled", "result")
    )
    assert dict(distribution) == pytest.approx(
        {
            (Fraction(0), Fraction(0), Fraction(1, 2)): 0.5,
            (Fraction(-1, 2), Fraction(-1), Fraction(-1, 2)): 0.5,
        },
        abs=1e-12,
    )
