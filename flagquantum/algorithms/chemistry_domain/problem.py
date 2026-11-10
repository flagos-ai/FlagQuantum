"""Real spatial-orbital integrals in one explicit chemistry convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...core.algebra.pauli import count, scalar
from ...observables.fermion import FermionOperator
from ..fermion.numerics import static_tensor
from ..molecular import MolecularHamiltonian, second_quantized_hamiltonian


@dataclass(frozen=True, slots=True)
class MolecularProblem:
    """Owned static integral snapshots for closed-shell spatial orbitals.

    ``two_body[p,q,r,s] = (pq|rs)`` in hartree. Spin orbitals interleave alpha
    and beta. Nuclear and frozen-core offsets are stored separately and enter
    ``fermion_hamiltonian`` exactly once. Tensor fields are owned snapshots,
    not deeply immutable arrays; callers must not mutate them in place.
    """

    one_body: torch.Tensor
    two_body: torch.Tensor
    n_electrons: int
    nuclear_repulsion: float = 0.0
    core_energy: float = 0.0
    orbital_labels: tuple[int, ...] = ()
    reference_energy: float | None = None
    reference_occupied: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        one = static_tensor(self.one_body, "one_body", ndim=2)
        two = static_tensor(self.two_body, "two_body", ndim=4)
        n = one.shape[0]
        if n < 1 or one.shape != (n, n) or two.shape != (n, n, n, n):
            raise ValueError(
                "integrals require square one-body and matching four-index tensors"
            )
        if float(one.imag.abs().max()) > 1e-12 or float(two.imag.abs().max()) > 1e-12:
            raise ValueError(
                "this spatial-integral contract supports real orbitals only"
            )
        one = one.real.contiguous()
        two = two.real.contiguous()
        if not torch.allclose(one, one.T, atol=1e-10, rtol=1e-10):
            raise ValueError("one_body must be symmetric")
        for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
            if not torch.allclose(
                two, two.permute(permutation), atol=1e-10, rtol=1e-10
            ):
                raise ValueError(
                    "two_body must obey real chemist-integral permutation symmetry"
                )
        electrons = count(self.n_electrons, "n_electrons")
        if electrons > 2 * n or electrons % 2:
            raise ValueError(
                "closed-shell electron count must be even and fit the orbital space"
            )
        labels = self.orbital_labels or tuple(range(n))
        labels = tuple(count(label, "orbital label") for label in labels)
        if len(labels) != n or len(set(labels)) != n:
            raise ValueError(
                "orbital labels must be distinct and match the orbital count"
            )
        occupied = (
            tuple(range(electrons // 2))
            if self.reference_occupied is None
            else tuple(
                count(q, "reference occupied orbital") for q in self.reference_occupied
            )
        )
        if (
            len(occupied) != electrons // 2
            or len(set(occupied)) != len(occupied)
            or any(q >= n for q in occupied)
        ):
            raise ValueError(
                "reference occupied orbitals must match the electron count and space"
            )
        for name in ("nuclear_repulsion", "core_energy", "reference_energy"):
            value = getattr(self, name)
            if value is None and name == "reference_energy":
                continue
            numeric = scalar(value)
            if numeric.imag:
                raise ValueError(f"{name} must be real")
            object.__setattr__(self, name, numeric.real)
        object.__setattr__(self, "one_body", one)
        object.__setattr__(self, "two_body", two)
        object.__setattr__(self, "n_electrons", electrons)
        object.__setattr__(self, "orbital_labels", labels)
        object.__setattr__(self, "reference_occupied", occupied)

    @property
    def n_orbitals(self) -> int:
        return int(self.one_body.shape[0])

    @property
    def n_modes(self) -> int:
        return 2 * self.n_orbitals

    @property
    def energy_offset(self) -> float:
        return self.nuclear_repulsion + self.core_energy

    @property
    def reference_occupation(self) -> tuple[int, ...]:
        """Return spin occupation bits even after arbitrary orbital reordering."""
        occupied = self.reference_occupied or ()
        return tuple(int(q // 2 in occupied) for q in range(self.n_modes))


def fermion_hamiltonian(problem: MolecularProblem) -> FermionOperator:
    """Assemble interleaved spin orbitals, including each energy offset once."""
    return second_quantized_hamiltonian(
        problem.one_body, problem.two_body, problem.energy_offset
    )


def from_native_molecule(molecule: MolecularHamiltonian) -> MolecularProblem:
    """Reuse the existing native chemistry path without changing its interface."""
    return MolecularProblem(
        molecule.one_body,
        molecule.two_body,
        molecule.n_electrons,
        molecule.nuclear_repulsion,
        molecule.core_energy,
        reference_energy=molecule.hf_energy,
    )
