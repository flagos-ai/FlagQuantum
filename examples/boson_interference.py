"""Two-photon interference and loss through the existing Lindblad namespace.

Run with ``python -m examples.boson_interference``. The model uses two explicitly
truncated modes, not a scalable photonic backend. Couplings and loss rates use
the inverse of the time-grid unit.
"""

from __future__ import annotations

import math

import torch

import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create


def main() -> None:
    coupling = 0.7
    hamiltonian = coupling * (create(0) * annihilate(1) + create(1) * annihilate(0))
    initial = torch.zeros(9, dtype=torch.complex128)
    initial[4] = 1  # |1, 1>, mode 0 most significant in dimensions=(3, 3).
    times = [0.0, math.pi / (4 * coupling)]

    for rate in (0.0, 0.2):
        plan = fql.plan(
            hamiltonian,
            initial,
            times,
            dimensions=(3, 3),
            collapse_operators=[(annihilate(0), rate), (annihilate(1), rate)],
            method="krylov-exponential",
        )
        restored = fql.LindbladPlan.from_json(plan.to_json())
        probabilities = fql.run(restored).populations[-1]
        print(f"Loss rate per mode: {rate:.1f}")
        print(f"P(0, 2): {probabilities[2].item():.8f}")
        print(f"P(2, 0): {probabilities[6].item():.8f}")
        print(f"P(1, 1): {abs(probabilities[4].item()):.8f}")
        print("Two-photon survival: " f"{probabilities[[2, 4, 6]].sum().item():.8f}")


if __name__ == "__main__":
    main()
