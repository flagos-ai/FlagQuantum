"""Two-photon interference and loss through the existing Lindblad namespace.

Run with ``python -m examples.boson_interference``. The model uses two explicitly
truncated modes, not a scalable photonic backend. Couplings and loss rates use
the inverse of the time-grid unit.
"""

from __future__ import annotations

import math

import torch

import flagquantum.lindblad as fql
from examples.terminal_instrument_reference import fock_reference
from flagquantum.observables.boson import annihilate, create
from flagquantum.runtime import sample_terminal


def interference_plan(loss_rate: float, *, cutoff: int = 3) -> fql.LindbladPlan:
    """Balanced interference of |1, 1>; compare three and four levels per mode."""
    if type(cutoff) is not int or cutoff not in (3, 4):
        raise ValueError("this reference compares three or four levels per mode")
    coupling = 0.7
    hamiltonian = coupling * (create(0) * annihilate(1) + create(1) * annihilate(0))
    initial = torch.zeros(cutoff**2, dtype=torch.complex128)
    initial[cutoff + 1] = 1  # |1, 1>, mode 0 most significant.
    times = [0.0, math.pi / (4 * coupling)]
    return fql.plan(
        hamiltonian,
        initial,
        times,
        dimensions=(cutoff, cutoff),
        collapse_operators=[(annihilate(0), loss_rate), (annihilate(1), loss_rate)],
        method="krylov-exponential",
        return_density_matrices=True,
    )


def counting_instrument() -> dict[str, tuple[torch.Tensor, ...]]:
    """Three-level QND number projectors in mode-0/mode-1 order."""
    _, single_mode = fock_reference()
    return {
        f"{left}/{right}": (torch.kron(a[0], b[0]),)
        for left, a in single_mode.items()
        for right, b in single_mode.items()
    }


def main() -> None:
    instrument = counting_instrument()
    for rate in (0.0, 0.2):
        plan = interference_plan(rate)
        evolution = fql.run(plan)
        restored = fql.LindbladPlan.from_json(plan.to_json())
        replay = fql.run(restored)
        probabilities = evolution.populations[-1]
        print(f"Loss rate per mode: {rate:.1f}")
        print(f"P(0, 2): {probabilities[2].item():.8f}")
        print(f"P(2, 0): {probabilities[6].item():.8f}")
        print(f"P(1, 1): {abs(probabilities[4].item()):.8f}")
        print("Two-photon survival: " f"{probabilities[[2, 4, 6]].sum().item():.8f}")
        if evolution.density_matrices is None or replay.density_matrices is None:
            raise RuntimeError(
                "number sampling requires the requested density matrices"
            )
        detected = sample_terminal(
            evolution.density_matrices[-1], instrument, shots=4096, seed=7
        )
        repeated = sample_terminal(
            replay.density_matrices[-1], instrument, shots=4096, seed=7
        )
        assert isinstance(detected.value, torch.Tensor)
        assert isinstance(repeated.value, torch.Tensor)
        print("Number labels:", detected.metadata["outcome_labels"])
        print("Number counts:", torch.bincount(detected.value, minlength=9).tolist())
        print("Shots:", detected.shots)
        print("Number sample replay:", torch.equal(detected.value, repeated.value))


if __name__ == "__main__":
    main()
