"""Minimize a circuit energy with no gradient and no random draw.

`flagquantum.algorithms.NelderMeadOptimizer` is reachable through the subpackage
surface only -- `from flagquantum.algorithms import NelderMeadOptimizer` --
because the algorithms package adds no root-level `fq.` name.

The premise, and both halves of it are concessions: this method decides
everything by comparing two objective values, so the objective has to be
deterministic, and the flag it reports at the end says a simplex collapsed onto a
local minimum rather than the global one. The first half is why this unit is not
a replacement for the SPSA unit beside it and the second is why it is not a
replacement for a global method: on a sampled objective the comparisons are
noise, and on a landscape with two wells the answer is a property of the starting
point. What it is for is the case the other units cannot serve at all -- an
objective that is exact, has no usable derivative, and must not be approximated.

Sizes, and why: two parameters on two qubits, the same instance
`docs/guides/ALGORITHMS.md` measures in its "Nelder-Mead simplex search"
section, and the SPSA comparison is run at the *same* evaluation budget rather
than the same step count, because a step costs the two units different amounts.
`docs/guides/ALGORITHMS.md` carries the unit's boundary.

Run it with:

    python -m examples.algorithms.nelder_mead_optimizer
"""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.algorithms import NelderMeadOptimizer, SPSAOptimizer
from flagquantum.errors import ValidationError

START = 0.4
MAXITER = 200
SPSA_PERTURBATION = 0.25
SPSA_SEEDS = (13, 5, 11)
NEAR_WELL_START = 0.5
GLOBAL_MINIMUM = -0.100617376638


def _energy(values: torch.Tensor) -> torch.Tensor:
    """Return the negative Pauli energy of a correlating two-qubit ansatz."""

    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()


def _two_well(values: torch.Tensor) -> torch.Tensor:
    """Two minima, the shallower one nearer the origin than the global one."""

    value = values[0]
    return ((value * value - 1.0) ** 2 + 0.1 * value).reshape(())


def _inplace(parameters: torch.Tensor) -> torch.Tensor:
    """An objective that writes into the tensor it is handed; this unit refuses it."""

    parameters.mul_(1.5)
    return (parameters**2).sum()


def _spsa_at_budget(steps: int, seed: int) -> tuple[float, int]:
    """Run SPSA for a fixed number of steps and return its energy and its calls."""

    optimizer = SPSAOptimizer(
        maxiter=steps,
        perturbation=SPSA_PERTURBATION,
        generator=torch.Generator().manual_seed(seed),
    )
    values = torch.full((2,), START, dtype=torch.float64)
    for _ in range(steps):
        values = optimizer.step(_energy, values)
    return float(_energy(values)), optimizer.evaluations


def main() -> None:
    start = torch.full((2,), START, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=MAXITER)
    result = optimizer.minimize(_energy, start)
    budget = result.evaluations

    spsa = [_spsa_at_budget(budget // 2, seed) for seed in SPSA_SEEDS]
    spsa_spread = max(value for value, _ in spsa) - min(value for value, _ in spsa)

    shallow = NelderMeadOptimizer(maxiter=400).minimize(
        _two_well, torch.tensor([NEAR_WELL_START], dtype=torch.float64)
    )

    refusals: list[tuple[str, str]] = []
    collinear = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    for label, call in (
        (
            "collinear simplex",
            lambda: NelderMeadOptimizer().minimize(
                _two_well, collinear[0], simplex=collinear
            ),
        ),
        (
            "in-place objective",
            lambda: NelderMeadOptimizer(maxiter=3).minimize(
                _inplace, torch.ones(2, dtype=torch.float64)
            ),
        ),
        ("budget", lambda: NelderMeadOptimizer(maxiter=0)),
        (
            "non-finite value",
            lambda: NelderMeadOptimizer(maxiter=3).minimize(
                lambda point: torch.tensor([float("inf")], dtype=torch.float64),
                torch.ones(2, dtype=torch.float64),
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
    print("Nelder-Mead simplex search -- flagquantum.algorithms.nelder_mead")
    print("=" * 72)
    print(f"  {'task':<24}: minimize a Pauli energy with no gradient, no")
    print(f"  {'':<24}  parameter-shift rule, and no random draw")
    print(f"  {'execution':<24}: local statevector on CPU")
    print(f"  {'premise':<24}: the objective has to be deterministic, and a")
    print(f"  {'':<24}  converged run reports a local minimum rather than the")
    print(f"  {'':<24}  global one")
    print()
    print("Nelder-Mead on the two-qubit energy")
    print(f"  {'start':<24}: {[START, START]}")
    print(f"  {'iterations':<24}: {result.iterations}")
    print(f"  {'objective calls':<24}: {result.evaluations}")
    print(f"  {'final energy':<24}: {result.value}")
    print(
        f"  {'final parameters':<24}: "
        f"{[round(float(v), 10) for v in result.parameters]}"
    )
    print(f"  {'converged':<24}: {result.converged}")
    print(f"  {'value spread':<24}: {result.value_spread:.3e}")
    print(f"  {'simplex spread':<24}: {result.simplex_spread:.3e}")
    print()
    print(f"the same objective at the same {budget}-evaluation budget")
    print(f"  {'Nelder-Mead':<24}: {result.value}")
    for seed, (value, calls) in zip(SPSA_SEEDS, spsa, strict=True):
        print(f"  {'SPSA, seed ' + str(seed):<24}: {round(value, 6)} in {calls} calls")
    print(f"  {'spread across seeds':<24}: {round(spsa_spread, 6)}")
    print()
    print("what converged does not mean, on a two-well quartic")
    print(f"  {'shallow-side start':<24}: [{NEAR_WELL_START}]")
    print(f"  {'reported value':<24}: {round(shallow.value, 12)}")
    print(f"  {'reported converged':<24}: {shallow.converged}")
    print(f"  {'simplex spread':<24}: {shallow.simplex_spread:.3e}")
    print(f"  {'global minimum':<24}: {GLOBAL_MINIMUM}")
    print(
        f"  {'distance from the global':<24}: {round(shallow.value - GLOBAL_MINIMUM, 12)}"
    )
    print()
    print("refusals, before a vertex is stored beside a value")
    for label, message in refusals:
        print(f"  {label:<20}: {message}")
    print()
    print("take away")
    print(
        f"  at {budget} evaluations Nelder-Mead lands on {result.value} exactly while"
    )
    print(
        f"  SPSA, given the same budget, lands within {round(spsa_spread, 6)} of the same"
    )
    print("  energy -- and its last digits are the seed rather than the optimum. That")
    print(
        "  is the whole reason both units exist: this one is exact on a deterministic"
    )
    print("  objective and defeated by a noisy one, and SPSA is the reverse. Neither")
    print("  is a global method, and the quartic above is the measurement of that for")
    print("  this one: it reports convergence a fifth of a unit above the global")
    print("  minimum, because the well it started nearest is not the deepest one.")


if __name__ == "__main__":
    main()
