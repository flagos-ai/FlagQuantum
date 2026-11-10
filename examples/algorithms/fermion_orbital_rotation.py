"""Differentiate a closed-shell H2 energy with respect to an orbital angle.

Run: python -m examples.algorithms.fermion_orbital_rotation
"""

from __future__ import annotations

import torch

from flagquantum.algorithms.chemistry_domain.orbital_rotation import (
    closed_shell_determinant_energy,
    rotate_spatial_integrals,
)
from flagquantum.algorithms.chemistry_domain.problem import from_native_molecule
from flagquantum.algorithms.molecular import create_molecular_hamiltonian


def main() -> None:
    print(
        "premise: fixed STO-3G hydrogen integrals; real CPU orbital rotations and "
        "a closed-shell determinant objective, not geometry gradients or SCF."
    )
    problem = from_native_molecule(
        create_molecular_hamiltonian([("H", (0, 0, 0)), ("H", (0, 0, 0.7414))])
    )
    generators = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]], dtype=torch.float64)

    def objective(angle: torch.Tensor) -> torch.Tensor:
        one, two = rotate_spatial_integrals(
            problem.one_body, problem.two_body, generators, angle
        )
        return closed_shell_determinant_energy(
            one, two, (0,), constant=problem.energy_offset
        )

    angle = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    energy = objective(angle)
    gradient = torch.autograd.grad(energy, angle)[0]
    step = 1e-6
    with torch.no_grad():
        finite_difference = (objective(angle + step) - objective(angle - step)) / (
            2 * step
        )
        initial = float(energy.detach())
        updated = float(objective(angle - 0.05 * gradient))
    torch.testing.assert_close(gradient[0], finite_difference, atol=1e-8, rtol=1e-7)
    assert abs(float(gradient[0])) > 1e-3
    assert updated < initial
    print(f"Determinant energy {initial:.10f} Ha")
    print(f"Orbital gradient {float(gradient[0]):.10f} Ha/rad")
    print(f"Finite difference {float(finite_difference):.10f} Ha/rad")
    print(f"After one gradient step {updated:.10f} Ha")


if __name__ == "__main__":
    main()
