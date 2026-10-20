"""Train three interval amplitudes through the existing Lindblad/PyTorch path.

Run: python -m examples.train_piecewise_drive
Expected output:
    Initial level-2 probability: 0.25121448
    Initial gradient: [0.04337116, 0.08674232, 0.02168558]
    Training reduced loss: True
"""

from __future__ import annotations

import torch

import flagquantum as fq
import flagquantum.lindblad as fql


def main() -> None:
    drive = fq.Parameter("drive")
    transition = torch.zeros((3, 3), dtype=torch.complex128)
    transition[1, 2] = transition[2, 1] = 1
    terms = [{"operator": transition, "coefficient": 0.5 * drive}]
    omega = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)

    def population() -> torch.Tensor:
        # Rebinding and rebuilding after each update keeps the plan's numerical
        # values and live graph aligned. JSON preserves values, not this graph.
        plan = fql.plan(
            terms,
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.3, 0.35],
            dimensions=(3,),
            parameter_values=[{"drive": value} for value in omega.unbind()],
            method="krylov-exponential",
            options=fq.ExecutionOptions(require_gradients=True),
        )
        return fql.run(plan).populations[-1, 2]

    initial = population()
    (gradient,) = torch.autograd.grad(initial, omega)
    initial_loss = (initial.detach() - 0.4) ** 2
    print(f"Initial level-2 probability: {initial.item():.8f}")
    print("Initial gradient:", [round(value, 8) for value in gradient.tolist()])
    optimizer = torch.optim.Adam([omega], lr=0.1)
    for _ in range(80):
        optimizer.zero_grad()
        loss = (population() - 0.4) ** 2
        loss.backward()
        optimizer.step()
    final_loss = (population().detach() - 0.4) ** 2
    print("Training reduced loss:", bool(final_loss < initial_loss / 100))


if __name__ == "__main__":
    main()
