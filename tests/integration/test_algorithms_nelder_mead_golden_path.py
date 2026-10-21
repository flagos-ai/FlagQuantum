from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The four refusals the example prints before a vertex is stored beside a
#: value. A collinear simplex cannot span the parameter directions, and an
#: objective that writes in place breaks the pairing the method depends on.
_REFUSALS = (
    "collinear simplex   : the vertices must span all 2 parameter directions",
    "in-place objective  : the objective modified the tensor it was given",
    "budget              : maxiter must be a positive integer",
    "non-finite value    : the objective returned a non-finite value",
)


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.algorithms.nelder_mead_optimizer"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_nelder_mead_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "Nelder-Mead simplex search -- flagquantum.algorithms.nelder_mead" in stdout
    assert "execution               : local statevector on CPU" in stdout
    assert "final energy            : -2.0" in stdout
    assert "converged               : True" in stdout


def test_the_deterministic_objective_is_where_this_unit_wins() -> None:
    """The comparison against SPSA is why both units exist.

    At one shared evaluation budget this method lands on the energy exactly
    while SPSA's last digits are its seed. The three SPSA seeds are printed so
    the spread is visible rather than asserted as a single number.
    """

    stdout = _run()

    assert "the same objective at the same 132-evaluation budget" in stdout
    assert "Nelder-Mead             : -2.0" in stdout
    for seed in ("seed 13", "seed 5", "seed 11"):
        assert f"SPSA, {seed}" in stdout, seed

    spread_line = next(
        line for line in stdout.splitlines() if "spread across seeds" in line
    )
    spread = float(spread_line.split(":")[1].strip())
    assert 0.0 < spread < 0.1, spread_line


def test_converged_reports_a_local_minimum_and_not_the_global_one() -> None:
    """`converged` measures a collapsing simplex, not an arriving energy.

    On a two-well quartic started from the shallow side the run reports
    convergence well above the global minimum, because the well it started
    nearest is not the deepest one. The distance is printed so the flag cannot
    be read as a correctness claim.
    """

    stdout = _run()

    assert "what converged does not mean, on a two-well quartic" in stdout
    assert "reported converged      : True" in stdout

    distance_line = next(
        line for line in stdout.splitlines() if "distance from the global" in line
    )
    distance = float(distance_line.split(":")[1].strip())
    assert distance > 0.1, distance_line


def test_the_example_refuses_before_storing_a_vertex() -> None:
    stdout = _run()

    for refusal in _REFUSALS:
        assert refusal in stdout, refusal
