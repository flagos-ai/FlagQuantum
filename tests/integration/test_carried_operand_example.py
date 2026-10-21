"""The carried-operand example, run as a user would run it.

The example is the only place the two routes a caller can take -- `Circuit.state`
and `run_dynamic` -- are printed side by side against the operand's own action, so
it is also where the claim "the operand executes as written" is checked as a
transcript rather than as an assertion about a helper. Every line pinned here is a
number the example computes; none of them is a timing or a host quantity.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

#: The operand's image of `|0>`, and the images the two named operators have of it.
#: The three lines are the whole comparison the example turns on: the first differs
#: from each of the others in a whole amplitude.
_OPERAND_IMAGE = "  operand |0>: +0.2800+0.0000j  +0.9600+0.0000j"
_X_IMAGE = "  x |0>      : +0.0000+0.0000j  +1.0000+0.0000j"
_Z_IMAGE = "  z |0>      : +1.0000+0.0000j  +0.0000+0.0000j"


def _run_example() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.carried_operand_execution"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_carried_operand_example_runs_end_to_end() -> None:
    stdout = _run_example()

    for line in (_OPERAND_IMAGE, _X_IMAGE, _Z_IMAGE):
        assert line in stdout, stdout

    # Four opcodes, each carrying the same operand. Each must produce the operand's
    # image on both routes, and the count below is why the block is one assertion
    # rather than sixteen: `x`, `y`, `z` and `rz` are the four names that had a
    # name-keyed kernel in front of the operand.
    for name in ("x", "y", "z", "rz"):
        carried = (
            f"{name} carrying the operand:\n"
            f"  circuit.state(): +0.2800+0.0000j  +0.9600+0.0000j\n"
            f"  planned instructions: 1\n"
            f"  run_dynamic(): +0.2800+0.0000j  +0.9600+0.0000j"
        )
        assert carried in stdout, f"{name}:\n{stdout}"


def test_the_example_shows_the_controls_that_must_not_change() -> None:
    """Without an operand the same opcode is the operator its name denotes.

    The example would pass its operand assertions with every fast path deleted, so
    the two controls are pinned: `x` on qubit 0 alone is `X|0>`, and a zero-angle
    `rz` that carries nothing is still removed by the identity pass.
    """

    stdout = _run_example()

    assert (
        "x with no operand:\n  circuit.state(): +0.0000+0.0000j  +1.0000+0.0000j"
        in stdout
    ), stdout
    assert "rz(0.0) with no operand:\n  planned instructions: 0" in stdout, stdout


def test_the_example_shows_the_declared_route_still_executing() -> None:
    """`any()` was always faithful, and the example keeps it as the reference.

    A reader can see that the four declared opcodes now agree with the route that
    never had a name-keyed fast path, which is what makes the change a repair
    rather than a new convention.
    """

    stdout = _run_example()

    assert (
        "the same operand on two qubits through any():\n"
        "  planned instructions: 1\n"
        "  circuit.state(): +0.078+0.000j  +0.269+0.000j  +0.269+0.000j  "
        "+0.922+0.000j" in stdout
    ), stdout
