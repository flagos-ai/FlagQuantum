"""Two finite-product models using the existing static CPU dynamics path.

Run with ``python -m examples.mixed_dimension_dynamics``. Frequencies and
collapse rates, if added, must use the inverse of the time-grid unit.
"""

from __future__ import annotations

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate


def main() -> None:
    times = torch.linspace(0.0, 0.5, 51, dtype=torch.float64)

    # A three-level model, with a leakage transition from level 1 to level 2.
    coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    transition = torch.zeros((3, 3), dtype=torch.complex128)
    transition[1, 2] = transition[2, 1] = 1
    leakage_plan = fql.plan(
        coupling * transition,
        [0.0, 1.0, 0.0],
        times,
        dimensions=(3,),
        options=fq.ExecutionOptions(require_gradients=True),
    )
    leakage = fql.run(leakage_plan).populations[-1, 2]
    (gradient,) = torch.autograd.grad(leakage, coupling)
    print(f"Three-level leakage probability: {leakage.item():.8f}")
    print(f"Coupling derivative: {gradient.item():.8f}")

    # Spin is the most significant subsystem; the mode keeps three Fock levels.
    lower = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    ladder = annihilate(0).to_matrix({0: 3})
    hamiltonian = 0.7 * (
        torch.kron(lower.T.contiguous(), ladder)
        + torch.kron(lower, ladder.T.contiguous())
    )
    initial = torch.zeros(6, dtype=torch.complex128)
    initial[3] = 1  # |spin=1, occupation=0>
    exchange_plan = fql.plan(hamiltonian, initial, times, dimensions=(2, 3))
    restored = fql.LindbladPlan.from_json(exchange_plan.to_json())
    exchange = fql.run(restored)
    print(f"Spin-mode dimensions: {restored.dimensions}")
    print(
        f"Probability of |spin=0, occupation=1>: {exchange.populations[-1, 1].item():.8f}"
    )


if __name__ == "__main__":
    main()
