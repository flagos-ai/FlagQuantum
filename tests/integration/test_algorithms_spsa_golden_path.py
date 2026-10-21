from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The four refusals the example prints before it runs an objective. SPSA reads
#: the objective twice per step and reports the cost at the unperturbed point,
#: so an objective that writes in place or returns a vector is refused by name
#: rather than silently costed against a tensor the optimizer no longer owns.
_REFUSALS = (
    "no gains      : SPSAOptimizer needs one of maxiter or stability",
    "negative c    : perturbation must be positive",
    "vector return : the objective must return exactly one value",
    "in-place write: the objective modified the tensor it was given",
)


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.algorithms.spsa_optimizer"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_spsa_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "SPSA optimization -- flagquantum.algorithms.spsa" in stdout
    assert "execution         : local statevector on CPU" in stdout

    # Both objectives the example drives have to report a descent. The energies
    # themselves are deliberately not pinned: the estimate is biased for every
    # finite perturbation, which the example states, so the digits are a
    # property of the seed and the platform's arithmetic rather than of SPSA.
    assert "convergence on the exact circuit energy" in stdout
    assert "convergence on a 4096-shot objective, which has no exact gradient" in (
        stdout
    )
    assert "generator seed    : 13" in stdout


def test_the_example_states_what_one_step_costs() -> None:
    """The two-evaluation cost is the whole reason the unit exists."""

    stdout = _run()

    # The comparison is the claim: the cost is two evaluations whatever the
    # parameter count, against four for the parameter-shift gradient on the
    # same two parameters.
    assert "parameter shift   : 4 circuit evaluations" in stdout
    assert "SPSA              : 2 circuit evaluations" in stdout


def test_the_example_refuses_before_running_the_objective() -> None:
    """A refused configuration must be named, not approximated."""

    stdout = _run()

    for refusal in _REFUSALS:
        assert refusal in stdout, refusal


def test_the_sampled_objective_reaches_the_exact_energy() -> None:
    """The shot-based half is what SPSA is for, so it must be measured.

    A 4096-shot objective has no exact gradient, so the run is scored by the
    exact energy at the parameter it stopped on rather than by the sampled
    value it read. The distance is printed and asserted as a bound rather than
    a value, because the sample is a draw.
    """

    stdout = _run()

    assert "exact energy there: -0.999" in stdout
    distance_line = next(
        line for line in stdout.splitlines() if "distance to -1" in line
    )
    distance = float(distance_line.split(":")[1].strip())
    assert distance < 1e-3, distance_line
