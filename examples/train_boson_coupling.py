"""Fit a static spin-mode exchange coupling with existing parameters and PyTorch.

Run with ``python -m examples.train_boson_coupling``. This finite CPU model uses
the first subsystem as a two-level mode and keeps three levels in the second.
It is not a hardware calibration or time-dependent control example.
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create


def main() -> None:
    exchange = create(0) * annihilate(1) + create(1) * annihilate(0)
    symbol = fq.Parameter("coupling")
    coupling = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    optimizer = torch.optim.Adam([coupling], lr=0.04)
    target = math.sin(0.7 * 0.5) ** 2
    losses: list[float] = []
    for _ in range(60):
        optimizer.zero_grad()
        # Binding reuses the circuit parameter vocabulary and keeps the tensor live.
        terms = [
            {"operator": exchange, "coefficient": symbol.bind({"coupling": coupling})}
        ]
        result = fql.run(
            terms,
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # |1, 0>, subsystem 0 most significant.
            [0.0, 0.5],
            dimensions=(2, 3),
            method="krylov-exponential",
            options=fq.ExecutionOptions(require_gradients=True),
        )
        loss = (result.populations[-1, 1] - target).square()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    print(f"Initial population loss: {losses[0]:.8f}")
    print(f"Final population loss: {losses[-1]:.8f}")
    print(f"Fitted coupling: {coupling.item():.8f}; reference: 0.70000000")


if __name__ == "__main__":
    main()
