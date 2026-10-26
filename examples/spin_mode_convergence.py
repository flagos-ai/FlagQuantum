"""Train and replay a resonant spin-mode model with controlled Fock truncation.

Spin is subsystem 0 (ground/excited), followed by a thermal oscillator with
p(n)=(1-q)q**n, q=0.2. Frequencies are angular frequencies in inverse time units.
The rotating-wave model is closed, resonant and lossless; it is not an ion-trap
pulse compiler, motional heating model or hardware calibration.
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create, number

THERMAL_RATIO = 0.2
FINAL_TIME = 0.7


def spin_mode_plan(cutoff: int, coupling: torch.Tensor) -> fql.LindbladPlan:
    """Build the same plan for convergence, differentiable fitting and replay."""
    if type(cutoff) is not int or cutoff < 2:
        raise ValueError("oscillator cutoff must be an integer >= 2")
    dimensions = {0: 2, 1: cutoff}
    exchange = create(0) * annihilate(1) + annihilate(0) * create(1)
    weights = THERMAL_RATIO ** torch.arange(cutoff, dtype=torch.float64)
    weights = weights / weights.sum()
    initial = torch.zeros((2 * cutoff, 2 * cutoff), dtype=torch.complex128)
    initial[cutoff:, cutoff:] = torch.diag(weights).to(torch.complex128)
    coefficient = fq.Parameter("coupling").bind({"coupling": coupling})
    return fql.plan(
        [{"operator": exchange, "coefficient": coefficient}],
        initial,
        [0.0, FINAL_TIME / 2, FINAL_TIME],
        dimensions=(2, cutoff),
        outputs=[
            fq.expectation(number(0).to_matrix(dimensions), name="excited"),
            fq.expectation((number(0) + number(1)).to_matrix(dimensions), name="total"),
        ],
        method="krylov-exponential",
        return_density_matrices=True,
        options=fq.ExecutionOptions(require_gradients=coupling.requires_grad),
    )


def reference_density(cutoff: int, coupling: float, time: float) -> torch.Tensor:
    """Independent closed Rabi blocks, including the artificial dark top level."""
    weights = [(1 - THERMAL_RATIO) * THERMAL_RATIO**n for n in range(cutoff)]
    normalization = sum(weights)
    density = torch.zeros((2 * cutoff, 2 * cutoff), dtype=torch.complex128)
    for n, weight in enumerate(weights):
        excited = cutoff + n
        if n == cutoff - 1:
            density[excited, excited] += weight / normalization
            continue
        angle = coupling * math.sqrt(n + 1) * time
        state = torch.zeros(2 * cutoff, dtype=torch.complex128)
        state[excited] = math.cos(angle)
        state[n + 1] = torch.tensor(-1j * math.sin(angle), dtype=torch.complex128)
        density += weight / normalization * torch.outer(state, state.conj())
    return density


def thermal_excited_probability(coupling: float, time: float) -> float:
    """Infinite-mode reference with omitted geometric tail below 0.2**64."""
    return sum(
        (1 - THERMAL_RATIO)
        * THERMAL_RATIO**n
        * math.cos(coupling * math.sqrt(n + 1) * time) ** 2
        for n in range(64)
    )


def main() -> None:
    coupling = torch.tensor(0.6, dtype=torch.float64)
    reference = thermal_excited_probability(coupling.item(), FINAL_TIME)
    for cutoff in (3, 5, 7):
        plan = spin_mode_plan(cutoff, coupling)
        result = fql.run(plan)
        assert result.density_matrices is not None
        expected = reference_density(cutoff, coupling.item(), FINAL_TIME)
        numerical_error = float(abs(result.density_matrices[-1] - expected).max())
        assert numerical_error < 1e-10
        total = result.expectation("total")
        assert float(abs(total - total[0]).max()) < 1e-10
        truncation_error = abs(float(result.expectation("excited")[-1]) - reference)
        # Conditioning away the tail costs <=q**N; the blocked top costs p(N-1).
        bound = THERMAL_RATIO**cutoff + (
            (1 - THERMAL_RATIO)
            * THERMAL_RATIO ** (cutoff - 1)
            / (1 - THERMAL_RATIO**cutoff)
        )
        assert truncation_error <= bound + 1e-10
        replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
        assert replay.density_matrices is not None
        assert torch.equal(result.density_matrices, replay.density_matrices)
        print(
            f"Cutoff {cutoff}: density error={numerical_error:.2e}, "
            f"thermal error={truncation_error:.2e}, bound={bound:.2e}"
        )
    fitted = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
    optimizer = torch.optim.Adam([fitted], lr=0.04)
    target = fql.run(spin_mode_plan(3, coupling)).expectation("excited")[-1].detach()
    losses: list[float] = []
    for _ in range(40):
        optimizer.zero_grad()
        result = fql.run(spin_mode_plan(3, fitted))
        loss = (result.expectation("excited")[-1] - target).square()
        torch.autograd.backward(loss)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] / 20
    numeric = spin_mode_plan(3, fitted.detach())
    original = fql.run(numeric).expectation("excited")
    replayed = fql.run(fql.LindbladPlan.from_json(numeric.to_json())).expectation(
        "excited"
    )
    assert torch.equal(original, replayed)
    print(f"Population loss: {losses[0]:.2e} -> {losses[-1]:.2e}")
    print("Total excitation conserved; exact numeric replay: True")


if __name__ == "__main__":
    main()
