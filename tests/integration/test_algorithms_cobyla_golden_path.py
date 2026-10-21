from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The four refusals the example prints before a value is stored beside a
#: point. COBYLA keeps a point and its function value together, so an
#: objective that writes in place is refused rather than stored against a
#: tensor that has since changed.
_REFUSALS = (
    "in-place objective  : the objective modified the tensor it was given",
    "budget              : maxiter must be a positive integer",
    "floor above the start: rhoend must be strictly below rhobeg",
    "non-finite constraint: constraint 0 returned a non-finite value",
)


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.algorithms.cobyla_optimizer"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_cobyla_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "COBYLA trust-region search -- flagquantum.algorithms.cobyla" in stdout
    assert "execution               : local statevector on CPU" in stdout
    assert "constraint              : angle 0 >= 0.9" in stdout


def test_the_constrained_run_stops_on_the_boundary_and_says_so() -> None:
    """A constrained run has to report feasibility beside its value.

    The bound is active at the optimum, so the run stops on it: the residual
    is zero, the point is feasible, and the energy it reports is the energy at
    the boundary rather than the one the same objective reaches unconstrained.
    Both numbers are printed so a reader can price the constraint.
    """

    stdout = _run()

    assert "violations              : [0.0]" in stdout
    assert "residual                : 0.0" in stdout
    assert "feasible                : True" in stdout
    assert "converged               : True" in stdout

    # The unconstrained run of the same objective from the same start is the
    # comparison that makes the constrained energy readable.
    assert "the same objective from the same start with no constraint" in stdout
    assert "final energy            : -2.0" in stdout


def test_the_example_shows_a_cheap_price_beating_the_feasible_optimum() -> None:
    """The failure mode is the point of the third case.

    The price on a violated constraint is a schedule rather than a calibrated
    weight, so a run at a cheap price can return an infeasible point whose
    value scores better than the feasible optimum's. Reading `value` alone
    would make that run look like the better answer; `feasible` beside it is
    what completes the report.
    """

    stdout = _run()

    assert "price 1.0               : [2.5, 2.5] value 0.5" in stdout
    assert "cheap feasible          : False" in stdout
    assert "price 2.0               : [2.0, 2.0] value 2.0" in stdout
    assert "exact feasible          : True" in stdout
    assert "a cheaper point scores  : 0.5 against the feasible optimum's 2.0" in stdout


def test_the_example_refuses_before_storing_a_value() -> None:
    stdout = _run()

    for refusal in _REFUSALS:
        assert refusal in stdout, refusal
