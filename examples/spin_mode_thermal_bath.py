"""Fit a finite spin-mode model in a fixed local thermal bath on CPU.

The rotating-wave exchange and inverse-time bath rates are phenomenological.
Blocked top-level upward flux is explicit; cutoff stability is not certification
of an infinite-mode density, ion-trap heating or a calibrated physical bath.
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create, number

# Frozen independent scalar E_ab reference: N=5, g=0.6, rates=(0.3,0.1), T=0.7.
TARGET_EXCITED = 0.8022880147390464


def thermal_bath_plan(
    cutoff: int,
    coupling: torch.Tensor,
    *,
    coherent: bool = False,
    rates: tuple[float, float] = (0.3, 0.1),
    steps: int = 2,
    method: str = "krylov-exponential",
) -> fql.LindbladPlan:
    """Compose fixed full-space jumps and the existing mixed-dimension planner.

    Examples:
        >>> plan = thermal_bath_plan(5, torch.tensor(0.6, dtype=torch.float64))
        >>> fql.run(plan).expectation("excited").shape
        torch.Size([3])
    """
    if type(cutoff) is not int or cutoff < 2:
        raise ValueError("oscillator cutoff must be an integer >= 2")
    if type(steps) is not int or steps < 2 or steps % 2:
        raise ValueError("steps must be an even integer >= 2")
    dimensions = {0: 2, 1: cutoff}
    exchange = create(0) * annihilate(1) + annihilate(0) * create(1)
    weights = 0.2 ** torch.arange(cutoff, dtype=torch.float64)
    weights = weights / weights.sum()
    spin = torch.tensor([1, 1j] if coherent else [0, 1], dtype=torch.complex128)
    if coherent:
        spin = spin / math.sqrt(2)
    initial = torch.kron(
        torch.outer(spin, spin.conj()), torch.diag(weights).to(torch.complex128)
    )
    down = annihilate(1).to_matrix(dimensions)
    top = torch.diag(
        torch.tensor(
            [int(n == cutoff - 1) for _ in range(2) for n in range(cutoff)],
            dtype=torch.complex128,
        )
    )
    coefficient = fq.Parameter("coupling").bind({"coupling": coupling})
    return fql.plan(
        [{"operator": exchange, "coefficient": coefficient}],
        initial,
        torch.arange(steps + 1, dtype=torch.float64) * (0.7 / steps),
        dimensions=(2, cutoff),
        collapse_operators=[
            {"operator": down, "rate": rates[0]},
            {"operator": down.mH, "rate": rates[1]},
        ],
        outputs=[
            fq.expectation(number(0).to_matrix(dimensions), name="excited"),
            fq.expectation(number(1).to_matrix(dimensions), name="mode"),
            fq.expectation(top, name="top"),
        ],
        method=method,
        return_density_matrices=True,
        options=fq.ExecutionOptions(require_gradients=coupling.requires_grad),
    )


def main() -> None:
    observations = {}
    for cutoff in (3, 5, 7, 9, 11):
        result = fql.run(
            thermal_bath_plan(cutoff, torch.tensor(0.6, dtype=torch.float64))
        )
        observations[cutoff] = torch.stack(
            [result.expectation(n) for n in ("excited", "mode", "top")], dim=-1
        )
        print(
            f"Cutoff {cutoff}: final spin/mode/top=",
            [round(float(value), 10) for value in observations[cutoff][-1]],
        )
    difference = abs(observations[9] - observations[11])
    assert float(difference[:, 0].max()) < 5e-5
    assert float(difference[:, 1].max()) < 1e-4
    assert float(observations[11][:, 2].max()) < 1e-5
    fitted = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
    optimizer = torch.optim.Adam([fitted], lr=0.04)
    initial_loss = 0.0
    for step in range(40):
        optimizer.zero_grad()
        prediction = fql.run(thermal_bath_plan(5, fitted)).expectation("excited")[-1]
        loss = (prediction - TARGET_EXCITED).square()
        if step == 0:
            initial_loss = float(loss.detach())
        torch.autograd.backward(loss)
        optimizer.step()
    plan = thermal_bath_plan(5, fitted.detach())
    original = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    final_loss = float((original.expectation("excited")[-1] - TARGET_EXCITED).square())
    assert final_loss <= 0.1 * initial_loss
    assert original.density_matrices is not None and replay.density_matrices is not None
    assert torch.equal(original.density_matrices, replay.density_matrices)
    print(f"Population objective: {initial_loss:.10f} -> {final_loss:.10f}")
    print(f"Fitted finite-model coupling: {float(fitted.detach()):.10f}")
    print("Cutoff stability checked separately; exact detached numeric replay: True")


if __name__ == "__main__":
    main()
