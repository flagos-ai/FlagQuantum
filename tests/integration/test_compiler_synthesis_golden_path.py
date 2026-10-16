from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The three synthesis cases the example drives, in the order it prints them.
#: The leaf counts and residuals are deliberately not pinned: the example checks
#: them itself, and the count depends on how a platform rounds an angle that the
#: short Euler forms compare exactly. The lines here are the ones a user reads.
_CASES = (
    "one-qubit synthesis",
    "two-qubit synthesis",
    "state-preparation synthesis",
)


def test_compiler_synthesis_user_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.compiler_synthesis"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "FlagQuantum compiler synthesis check passed" in completed.stdout
    assert "target basis: z-rotation 'rz', pi/2 pulse 'sx', entangler 'cx'" in (
        completed.stdout
    )

    for case in _CASES:
        assert case in completed.stdout, case

    # Each case proves its own rewrite against the shipped runtime, so one
    # overlap line per case is what makes the run evidence rather than a print.
    assert completed.stdout.count("|overlap| = ") == len(_CASES)
    assert completed.stdout.count("every leaf is in the target basis: True") == len(
        _CASES
    )

    # The boundary's refusals are half of what a contributor needs to know.
    assert "one-qubit z_rotation='u3'      -> None" in completed.stdout
    assert "one-qubit pulse_opcode='ry'    -> None" in completed.stdout
    assert "ladder    z_rotation='phase'   -> None" in completed.stdout
    assert "two-qubit entangler='ecr'      -> None" in completed.stdout


def test_the_example_refuses_to_approximate() -> None:
    """The refusals are the example's, not the test's, so they must be printed."""

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.compiler_synthesis"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.count("-> None") == 4
    assert "but the one-qubit path does accept 'phase'" in completed.stdout
