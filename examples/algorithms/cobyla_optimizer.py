"""Minimize a circuit energy under a constraint, and read the flag it returns.

`flagquantum.algorithms.CobylaOptimizer` is reachable through the subpackage
surface only -- `from flagquantum.algorithms import CobylaOptimizer` -- because
the algorithms package adds no root-level `fq.` name.

The premise has two halves, and the first is the reason this unit is not a
replacement for either optimizer beside it: the subproblem it solves at every
step is a *linearization*, so a step is only as good as the affine models are at
the current radius, and a run that converges reports that its trust region
reached its floor rather than that it found a constrained optimum. The second
half is what a caller has to do about that: the price put on a violated
constraint is a schedule and not a calibrated weight, so a run can return a point
that is infeasible and scores better than the feasible optimum, and `feasible`
is reported beside `value` precisely so that misreading is visible rather than
silent. Both halves are measured below.

Sizes, and why: two parameters on two qubits, and the constrained energy is the
same instance `docs/guides/ALGORITHMS.md` measures in its "COBYLA trust-region
search" section. The price comparison is run on one quadratic with one affine
constraint, changing only the starting price, because that is the smallest
instance on which the second half of the premise is a number rather than a
warning. `docs/guides/ALGORITHMS.md` carries the unit's boundary.

Run it with:

    python -m examples.algorithms.cobyla_optimizer
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.errors import ValidationError

START = 0.2
ANGLE_FLOOR = 0.9
MAXITER = 400
SCORE_TARGET = (3.0, 3.0)
CHEAP_PRICE = 1.0
EXACT_PRICE = 2.0


def _energy(values: torch.Tensor) -> torch.Tensor:
    """Return the negative Pauli energy of a correlating two-qubit ansatz."""

    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()


def _score(values: torch.Tensor) -> torch.Tensor:
    """A separable quadratic whose unconstrained minimum is (3, 3)."""

    shifts = torch.tensor(SCORE_TARGET, dtype=values.dtype)
    return ((values - shifts) ** 2).sum()


def _budget(values: torch.Tensor) -> torch.Tensor:
    """The one affine constraint both price runs share: x + y <= 4."""

    return 4.0 - values[0] - values[1]


def _angle_floor(values: torch.Tensor) -> torch.Tensor:
    """The constraint that pushes the energy off its unconstrained minimum."""

    return values[0] - ANGLE_FLOOR


def _inplace(parameters: torch.Tensor) -> torch.Tensor:
    """An objective that writes into the tensor it is handed; this unit refuses it."""

    parameters.mul_(1.5)
    return (parameters**2).sum()


def main() -> None:
    start = torch.full((2,), START, dtype=torch.float64)

    bounded = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8, maxiter=MAXITER).minimize(
        _energy, start, constraints=[_angle_floor]
    )
    unconstrained = CobylaOptimizer(rhobeg=0.5, rhoend=1e-8, maxiter=MAXITER).minimize(
        _energy, start
    )

    origin = torch.zeros(2, dtype=torch.float64)
    cheap = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty=CHEAP_PRICE
    ).minimize(_score, origin, constraints=[_budget])
    exact = CobylaOptimizer(
        rhobeg=0.5, rhoend=1e-10, maxiter=500, penalty=EXACT_PRICE
    ).minimize(_score, origin, constraints=[_budget])

    refusals: list[tuple[str, str]] = []
    for label, call in (
        (
            "in-place objective",
            lambda: CobylaOptimizer(maxiter=3).minimize(
                _inplace, torch.ones(2, dtype=torch.float64)
            ),
        ),
        ("budget", lambda: CobylaOptimizer(maxiter=0)),
        (
            "floor above the start",
            lambda: CobylaOptimizer(rhobeg=0.5, rhoend=1.0),
        ),
        (
            "non-finite constraint",
            lambda: CobylaOptimizer(maxiter=3).minimize(
                lambda point: point.sum(),
                torch.ones(2, dtype=torch.float64),
                constraints=[lambda point: torch.tensor(float("inf"))],
            ),
        ),
    ):
        try:
            call()
        except ValidationError as error:
            refusals.append((label, str(error).split(",")[0]))
        else:
            raise AssertionError(f"{label} stopped being refused")

    print("=" * 72)
    print("COBYLA trust-region search -- flagquantum.algorithms.cobyla")
    print("=" * 72)
    print(f"  {'task':<24}: minimize a constrained circuit energy with no")
    print(f"  {'':<24}  gradient and no random draw")
    print(f"  {'execution':<24}: local statevector on CPU")
    print(f"  {'premise':<24}: the subproblem is a linearization, so a converged")
    print(f"  {'':<24}  run reports that its trust region reached its floor")
    print(f"  {'':<24}  rather than that it found a constrained optimum, and")
    print(f"  {'':<24}  the price it puts on a violated constraint is a")
    print(f"  {'':<24}  schedule rather than a calibrated weight, so a run can")
    print(f"  {'':<24}  return an infeasible point that scores better than the")
    print(f"  {'':<24}  feasible optimum")
    print()
    print("the two-qubit energy under a bound on one angle")
    print(f"  {'constraint':<24}: angle 0 >= {ANGLE_FLOOR}")
    print(f"  {'start':<24}: {[START, START]}")
    print(f"  {'iterations':<24}: {bounded.iterations}")
    print(f"  {'objective calls':<24}: {bounded.evaluations}")
    print(f"  {'final energy':<24}: {bounded.value}")
    print(
        f"  {'final parameters':<24}: "
        f"{[round(float(v), 10) for v in bounded.parameters]}"
    )
    print(f"  {'violations':<24}: {[round(float(v), 10) for v in bounded.violations]}")
    print(f"  {'residual':<24}: {bounded.residual}")
    print(f"  {'feasible':<24}: {bounded.feasible}")
    print(f"  {'rho':<24}: {bounded.rho:.3e}")
    print(f"  {'rhoend':<24}: {bounded.rhoend:.3e}")
    print(f"  {'converged':<24}: {bounded.converged}")
    print(f"  {'final price':<24}: {bounded.penalty}")
    print()
    print("the same objective from the same start with no constraint")
    print(f"  {'final energy':<24}: {unconstrained.value}")
    print(
        f"  {'final parameters':<24}: "
        f"{[round(float(v), 10) for v in unconstrained.parameters]}"
    )
    print(f"  {'converged':<24}: {unconstrained.converged}")
    print(
        f"  {'energy the bound costs':<24}: "
        f"{round(bounded.value - unconstrained.value, 10)}"
    )
    print(
        f"  {'exact value at the floor':<24}: {round(-2.0 * math.cos(ANGLE_FLOOR), 10)}"
    )
    print(
        f"  {'energy the bound leaves':<24}: "
        f"{round(bounded.value + 2.0 * math.cos(ANGLE_FLOOR), 12)}"
    )
    print()
    print("what the price decides, on one quadratic under x + y <= 4")
    print(
        f"  {'unconstrained minimum':<24}: "
        f"{[round(float(v), 6) for v in SCORE_TARGET]}"
    )
    print(
        f"  {'price ' + str(CHEAP_PRICE):<24}: "
        f"{[round(float(v), 6) for v in cheap.parameters]} "
        f"value {round(cheap.value, 6)}"
    )
    print(f"  {'cheap residual':<24}: {round(cheap.residual, 6)}")
    print(f"  {'cheap feasible':<24}: {cheap.feasible}")
    print(f"  {'cheap final price':<24}: {cheap.penalty}")
    print(
        f"  {'price ' + str(EXACT_PRICE):<24}: "
        f"{[round(float(v), 6) for v in exact.parameters]} "
        f"value {round(exact.value, 6)}"
    )
    print(f"  {'exact residual':<24}: {round(exact.residual, 6)}")
    print(f"  {'exact feasible':<24}: {exact.feasible}")
    print(f"  {'exact final price':<24}: {exact.penalty}")
    print(
        f"  {'a cheaper point scores':<24}: "
        f"{round(cheap.value, 6)} against the feasible optimum's "
        f"{round(exact.value, 6)}"
    )
    print()
    print("refusals, before a value is stored beside a point")
    for label, message in refusals:
        print(f"  {label:<20}: {message}")
    print()
    print("take away")
    print(
        f"  the run stops exactly on the boundary at {round(float(bounded.parameters[0]), 10)}"
    )
    print(
        f"  and reports an energy {round(bounded.value, 6)}, which is the energy there"
    )
    print(f"  rather than the -2.0 the same objective reaches with no constraint. That")
    print(
        f"  part is the method working. The part that is not: at a price of {CHEAP_PRICE} the"
    )
    print(
        f"  same objective under the same constraint returns {round(cheap.value, 6)} at"
    )
    print(
        f"  {[round(float(v), 6) for v in cheap.parameters]}, which violates the constraint by"
    )
    print(
        f"  {round(cheap.residual, 6)} and is a better score than the feasible optimum's"
    )
    print(
        f"  {round(exact.value, 6)}. Read `value` alone and that run looks like the better"
    )
    print(
        f"  answer. Read `feasible` beside it and the report is complete: a local method,"
    )
    print("  a scheduled price, and a residual that says which one you got.")


if __name__ == "__main__":
    main()
