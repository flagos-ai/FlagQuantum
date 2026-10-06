"""Minimize a circuit energy with two evaluations per step, at demonstration scale.

`flagquantum.algorithms.SPSAOptimizer` is reachable through the subpackage
surface only -- `from flagquantum.algorithms import SPSAOptimizer` -- because the
algorithms package adds no root-level `fq.` name.

The premise, and it is a concession: the update SPSA applies is built from a
finite-difference estimate, which is an estimate rather than a gradient, and its
expectation is the gradient only as the perturbation shrinks. For an objective
that already has an exact gradient, reverse-mode autograd is cheaper and exact
without it, and this script measures both costs rather than asserting either.
SPSA is worth its bias on the case that has no other option: an objective whose
only accessible value is a sample.

Sizes, and why: two parameters on two qubits, which is the instance
`docs/guides/ALGORITHMS.md` measures in its "SPSA optimization" section, and 4096
shots for the sampled objective. The cost comparison runs a real parameter-shift
gradient, one shift pair per parameter, so its evaluation count is counted rather
than derived. `docs/guides/ALGORITHMS.md` carries the unit's boundary.

Run it with:

    python -m examples.algorithms.spsa_optimizer
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
from flagquantum.algorithms import SPSAOptimizer
from flagquantum.errors import ValidationError

SEED = 13
STEPS = 120
PERTURBATION = 0.25
START = 0.4
SHOTS = 4096
SAMPLED_SEED = 17
SAMPLED_STEPS = 200
SHIFT = math.pi / 2.0


def _energy(values: torch.Tensor) -> torch.Tensor:
    """Return the negative Pauli energy of a correlating two-qubit ansatz."""
    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
    outputs = fq.expectation(fq.Z(0) + fq.Z(1))
    return -fq.run(circuit, outputs=outputs).expectation().sum()


def _parameter_shift_gradient(values: torch.Tensor, calls: list[int]) -> torch.Tensor:
    """Return the exact gradient, counting the circuit evaluations it spends."""
    gradient = torch.empty_like(values)
    for index in range(values.numel()):
        for sign in (1.0, -1.0):
            shifted = values.clone()
            shifted[index] += sign * SHIFT
            calls[0] += 1
            gradient[index] = _energy(shifted)
        gradient[index] /= 2.0 * math.sin(SHIFT)
    return gradient


def _inplace(parameters: torch.Tensor) -> torch.Tensor:
    """An objective that writes into the tensor it is handed; SPSA refuses it."""
    parameters.mul_(1.5)
    return (parameters**2).sum()


def _sampled_energy(counter: list[int]) -> object:
    """Return a single-qubit objective whose value is a 4096-shot sample."""

    def objective(values: torch.Tensor) -> torch.Tensor:
        counter[0] += 1
        circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, values[0])
        result = fq.run(
            circuit,
            outputs=fq.samples([0]),
            options=fq.ExecutionOptions(shots=SHOTS, seed=counter[0]),
        )
        return -(1.0 - 2.0 * result.samples.to(torch.float64).mean())

    return objective


def main() -> None:
    start = torch.full((2,), START, dtype=torch.float64)

    shift_calls = [0]
    exact_gradient = _parameter_shift_gradient(start, shift_calls)

    optimizer = SPSAOptimizer(
        maxiter=STEPS,
        perturbation=PERTURBATION,
        generator=torch.Generator().manual_seed(SEED),
    )
    values = start.clone()
    initial = float(_energy(values))
    for _ in range(STEPS):
        values = optimizer.step(_energy, values)
    final = float(_energy(values))

    sampled_calls = [0]
    sampled = _sampled_energy(sampled_calls)
    sampler = SPSAOptimizer(
        maxiter=SAMPLED_STEPS,
        perturbation=0.3,
        generator=torch.Generator().manual_seed(SAMPLED_SEED),
    )
    shot_values = torch.tensor([0.35], dtype=torch.float64)
    shot_initial = float(sampled(shot_values))
    for _ in range(SAMPLED_STEPS):
        shot_values = sampler.step(sampled, shot_values)
    shot_exact = -math.cos(float(shot_values[0]))

    refusals: list[tuple[str, str]] = []
    for label, call in (
        ("no gains", lambda: SPSAOptimizer()),
        ("negative c", lambda: SPSAOptimizer(maxiter=10, perturbation=-0.1)),
        (
            "vector return",
            lambda: SPSAOptimizer(maxiter=10).estimate_gradient(
                lambda p: p * 2.0, torch.tensor([1.0, 2.0])
            ),
        ),
        (
            "in-place write",
            lambda: SPSAOptimizer(maxiter=10).estimate_gradient(
                _inplace, torch.tensor([1.0, 2.0])
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
    print("SPSA optimization -- flagquantum.algorithms.spsa")
    print("=" * 72)
    print(f"  {'task':<18}: minimize a Pauli energy with two evaluations per step,")
    print(f"  {'':<18}  and from 4096-shot samples of the same energy")
    print(f"  {'execution':<18}: local statevector on CPU")
    print(
        f"  {'premise':<18}: the estimate is an estimate rather than a gradient, and an"
    )
    print(f"  {'':<18}  objective with an exact gradient is served cheaper and exact")
    print(f"  {'':<18}  without it")
    print()
    print("cost of one descent step on 2 parameters")
    print(f"  {'parameter shift':<18}: {shift_calls[0]} circuit evaluations")
    print(f"  {'SPSA':<18}: {optimizer.evaluations // STEPS} circuit evaluations")
    print(f"  {'exact gradient':<18}: {[round(float(g), 6) for g in exact_gradient]}")
    print()
    print("convergence on the exact circuit energy")
    print(f"  {'start':<18}: {[START, START]}")
    print(f"  {'perturbation':<18}: {PERTURBATION}")
    print(f"  {'generator seed':<18}: {SEED}")
    print(f"  {'initial energy':<18}: {round(initial, 6)}")
    print(f"  {'final energy':<18}: {round(final, 6)}")
    print(f"  {'final parameters':<18}: {[round(float(v), 6) for v in values]}")
    print(f"  {'objective calls':<18}: {optimizer.evaluations}")
    print()
    print(f"convergence on a {SHOTS}-shot objective, which has no exact gradient")
    print(f"  {'start':<18}: [0.35]")
    print(f"  {'initial sample':<18}: {round(shot_initial, 6)}")
    print(f"  {'final parameter':<18}: {round(float(shot_values[0]), 6)}")
    print(f"  {'exact energy there':<18}: {round(shot_exact, 6)}")
    print(f"  {'distance to -1':<18}: {abs(shot_exact + 1.0):.3e}")
    print(f"  {'sampled calls':<18}: {sampled_calls[0]}")
    print()
    print("refusals, before the objective runs")
    for label, message in refusals:
        print(f"  {label:<14}: {message}")
    print()
    print("take away")
    print("  the estimate costs two evaluations whatever the parameter count, which is")
    print(
        f"  what SPSA is for; the parameter-shift gradient above costs {shift_calls[0]}, and"
    )
    print(
        "  that is one step's worth of a gradient optimizer. The estimate is biased for"
    )
    print(
        "  every finite perturbation, so the last digits of the trajectory are the draw"
    )
    print(
        "  and not the optimum: seed 13 lands at",
        round(final, 6),
        "where seeds 5 and 11",
    )
    print("  land at -1.999182 and -1.999094 on the same run length.")


if __name__ == "__main__":
    main()
