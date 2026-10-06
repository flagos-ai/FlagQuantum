"""Hartree-Fock, the molecular-orbital basis, and the qubit Hamiltonian.

:mod:`flagquantum.algorithms.chemistry_integrals` evaluates the integrals of a
molecule in its atomic-orbital basis. Those integrals are not yet a chemistry
workload: a variational eigensolver needs canonical molecular orbitals, a
reference energy, and a Hamiltonian whose operator algebra is already expressed
in the basis the ansatz prepares. That is what this module adds, and it is the
second half of the entry point CUDA-Q calls
``cudaq.chemistry.create_molecular_hamiltonian``.

**The thesis: the driver is a sequence of closed forms, not a package.**
CUDA-Q's chemistry entry point is a shim. Its Python half imports
``openfermion`` and ``openfermionpyscf``, hands the geometry to
``openfermionpyscf.run_pyscf``, and maps the result through
``openfermion.jordan_wigner``; its C++ half resolves a ``MoleculePackageDriver``
out of a registry keyed on the string ``"pyscf"`` and ignores the ``driver``
argument it was handed. The work behind that shim -- an SCF iteration, an
orbital transformation, an active-space selection, a second-quantised
Hamiltonian -- is arithmetic over the integrals this repository already has, so
it is performed here instead. There is no ``driver`` argument to pass and none
to ignore: this module *is* the driver, and it imports nothing beyond ``torch``,
which is this package's declared core requirement.

**Where the spin-orbital Hamiltonian's index pattern comes from.** The
second-quantised electronic Hamiltonian in a spatial-orbital basis is

    H = sum_{pq} h_{pq} a^dag_{p,s} a_{q,s}
        + 1/2 sum_{pqrs} sum_{s,t} (ps|qr) a^dag_{p,s} a^dag_{q,t} a_{s,t} a_{r,s}

with ``s`` and ``t`` the spin labels and ``(ps|qr)`` the integral in the chemist
convention. Expanding the spin sum leaves exactly four patterns, which are the
four ways the first creation can share a spin with the second annihilation while
the second creation shares a spin with the first annihilation. The stored array
is ``two_body[p, q, r, s] = (pq|rs)``, so the integral the pattern needs is
``two_body[p, r, q, s]`` and the monomial is ``a^dag_P a^dag_Q a_S a_R`` -- the
annihilations in the reverse of the order the integral's indices are read. That
reversal is not a convention to be checked for tidiness; it is checked by
physics. Assembling over ``two_body[p, s, q, r]`` instead moves the Hartree-Fock
determinant's expectation from -74.9650107063 to -118.3298603077 for water and
from -7.8620238601 to -9.9855034635 for lithium hydride, while the pattern used
here reproduces both SCF energies.

**Why the operator handed to the qubit mapping is symmetrised first.** The
assembled coefficients satisfy ``c(a^dag_P a_Q) = conj(c(a^dag_Q a_P))`` only to
the assembly's own rounding: the largest such defect is 2.546e-15 over the eight
reference molecules below, and it is nonzero. :func:`flagquantum.observables.jordan_wigner`
refuses a non-Hermitian operator by name rather than dropping the imaginary part
of a coefficient, so the constructor replaces the assembled operator with its
exact Hermitian part ``(H + H^dag) / 2``. The replacement is a projection whose
distance from the assembled operator is that same 2.5e-15.

**What has been measured, rather than asserted.** Every number below was
produced by this module's own pipeline at the standard geometries of the eight
closed-shell molecules the STO-3G table covers, and the reference energies are
the restricted Hartree-Fock values in that basis.

* The Hartree-Fock energy reproduces the reference to at worst 4.45e-11 (N2).
  The density change at the stopping iteration is below the 1e-10 tolerance by
  construction and the SCF residual ``max |F P S - S P F|`` is at most 4.8e-11.
* The orbital energies and the converged density satisfy the exact identity
  ``E = sum_i (h_ii + eps_i) + E_nuc`` over the occupied orbitals to at worst
  1.8e-10 (CO), which is a statement about the integrals and the solve together
  rather than about either alone.
* The Hartree-Fock determinant's expectation ``<HF|H|HF> + E_nuc`` equals the SCF
  energy to at worst 8.53e-14, and the same value read off the diagonal of the
  Jordan-Wigner image of the Hamiltonian agrees to 2.84e-14. This is the check
  that pins the index pattern above.
* The assembled Hamiltonian is a functional of the integrals alone, so it is
  invariant under a rotation of the orbitals: applying a random orthogonal mixing
  to ``one_body`` and ``two_body`` together moves the exact ground state by at
  most 7.11e-14 (H2O).
* The frozen-core reduction is exact rather than approximate: folding the lowest
  orbital into a constant and an effective one-body term and diagonalising the
  active space gives the same energy as diagonalising the full Hamiltonian
  restricted to determinants with that orbital doubly occupied, to 1.78e-15
  (LiH) and 0.00e+00 (H2O).
* The exact ground state of the H2 Hamiltonian at 0.7414 angstrom is
  -1.1372701747, which is the published full-configuration-interaction value for
  that basis, -1.137270.

**What is not here.** Restricted closed-shell Hartree-Fock is the only solve, so
a multiplicity other than one and an odd electron count are refused rather than
approximated by a solution of the wrong symmetry. The exact ground state is
available only inside :data:`MAX_FCI_DETERMINANTS`; outside it the field is
``None`` rather than an approximation to it. There is no geometry optimisation,
no gradient, no symmetry or point-group detection, no atomisation energy and no
basis-set extrapolation; STO-3G is the only basis, because it is the only one
this repository evaluates. The frozen orbitals are assumed to be the occupied
canonical orbitals of lowest energy, which is what the solve returns, and the
module does not check that they are the ones a caller would want frozen.

**How this meets the ansatz half.**
:func:`flagquantum.algorithms.chemistry.uccsd_excitations` takes an electron count
and a spin-orbital count and fills the lowest spin orbitals for its reference
determinant, which is exactly the occupation :attr:`MolecularHamiltonian.n_electrons`
and :attr:`MolecularHamiltonian.n_orbitals` describe. The two halves meet with no
conversion in between.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from ..errors import CapabilityError, ValidationError
from ..observables import Observable, jordan_wigner
from ..observables.fermion import FermionOperator, FermionTerm, annihilate, create
from .chemistry_integrals import (
    MolecularGeometry,
    MolecularIntegrals,
    molecular_integrals,
)

__all__ = (
    "MAX_FCI_DETERMINANTS",
    "MOLECULAR_ASSUMPTIONS",
    "MOLECULAR_LIMITATIONS",
    "HartreeFockSolution",
    "MolecularHamiltonian",
    "create_molecular_hamiltonian",
    "exact_ground_state_energy",
    "restricted_hartree_fock",
)

#: The largest determinant space this module diagonalises exactly.
#:
#: The space is ``C(n_orbitals, n_electrons / 2) ** 2``. At the standard
#: geometries of the eight closed-shell molecules the STO-3G table covers, it is
#: 4 (H2), 36 (HF), 225 (LiH), 441 (H2O), 3136 (NH3), 14400 (N2), 14400 (CO) and
#: 15876 (CH4); the first four are inside the bound and the last four are not.
MAX_FCI_DETERMINANTS = 1024

#: The spin patterns the two-electron operator's spin sum leaves behind.
_SPIN_PATTERNS = ((0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0))

_SCF_TOLERANCE = 1e-10
_SCF_MAX_ITERATIONS = 400

MOLECULAR_ASSUMPTIONS: tuple[str, ...] = (
    "Hartree-Fock is restricted and closed-shell. The solve takes one doubly "
    "occupied spatial orbital per electron pair, so it is a statement about a "
    "singlet and about nothing else; an open-shell ground state is refused rather "
    "than approximated by a solution of the wrong symmetry.",
    "The energy the solve converges to is stationary in the density, not proven to "
    "be the lowest. The iteration starts from the bare core Hamiltonian and moves "
    "to the fixed point nearest it, and a second solution at a different energy is "
    "not searched for. Every reference energy below is reproduced, which is "
    "evidence that the fixed point reached is the intended one for these eight "
    "molecules and not that no other fixed point exists.",
    "The frozen-core reduction is exact for the active space it defines. The frozen "
    "orbitals are held doubly occupied, so the eigenvalues of the reduced "
    "Hamiltonian are the eigenvalues of the full Hamiltonian restricted to those "
    "determinants -- a subspace, not an approximation to the full space. An active "
    "space that also drops virtual orbitals is not a subspace and is an "
    "approximation.",
)

MOLECULAR_LIMITATIONS: tuple[str, ...] = (
    "STO-3G is the only basis, because it is the only basis this repository "
    "evaluates. A larger basis changes the energies reported here, and comparing a "
    "value from this module with a value from a correlation-consistent basis "
    "compares two different Hamiltonians.",
    "The exact ground state is dense diagonalisation of the S_z = 0 sector, so it "
    "is available only at or below MAX_FCI_DETERMINANTS and is None above it. A "
    "molecule outside that bound still gets a Hamiltonian, an SCF energy and a "
    "Hartree-Fock determinant; what it does not get is an energy to compare an "
    "eigensolver against.",
    "The one- and two-electron integrals are dense and unscreened, so the orbital "
    "transformation costs the fourth power of the orbital count and the assembled "
    "operator carries every quadruple whose integral is nonzero. Nothing here is "
    "intended for a basis much larger than STO-3G over H to Ne.",
    "An active space is a contiguous window of canonical orbitals starting "
    "immediately above the frozen core. Selecting orbitals by symmetry, by "
    "occupation or by an entropy criterion is not offered, and neither is a check "
    "that the window a caller asks for is the one a chemistry problem wants.",
    "fci_energy is the ground state of the Hamiltonian this call returns. When an "
    "active space drops occupied or virtual orbitals that is the active-space "
    "value, which lies above the full-space value; CUDA-Q's fci_energy comes from "
    "PySCF's full-space FCI and does not move with the active space, so the two "
    "fields diverge by construction once an active space is requested.",
    "The undamped iteration has a two-cycle at a stretched bond and refuses there "
    "rather than converging. For H2 in this basis it converges in two steps out to "
    "5 A and oscillates with a density change of exactly 2.0 at 10 A and beyond, "
    "where the HOMO and LUMO are degenerate to the arithmetic floor and the bare "
    "core Hamiltonian start is as close to one solution as to the other. The "
    "refusal is the CapabilityError above; damping the step would converge it and "
    "is not implemented, so a caller with a stretched bond gets a named failure "
    "instead of a density that is one iteration from either fixed point.",
)


@dataclass(frozen=True, slots=True)
class HartreeFockSolution:
    """A converged restricted Hartree-Fock solve.

    ``coefficients`` is the ``(n, n)`` matrix whose columns are the canonical
    molecular orbitals in the atomic-orbital basis that was handed in, so
    ``coefficients[:, i]`` is orbital ``i``, and ``density`` is the
    one-particle density matrix ``2 C_occ C_occ^T``. ``density_change`` and
    ``residual`` are the two quantities the iteration measured when it stopped:
    the largest change in a density-matrix element between the last two
    iterations, and ``max |F P S - S P F|``, the amount by which the converged
    density fails to commute with the Fock matrix.

    Examples:
        >>> from flagquantum.algorithms.chemistry_integrals import molecular_integrals
        >>> from flagquantum.algorithms.molecular import restricted_hartree_fock
        >>> integrals = molecular_integrals(
        ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        ... )
        >>> solve = restricted_hartree_fock(integrals, 2)
        >>> round(solve.energy, 10)
        -1.1166843871
        >>> solve.n_iterations
        2
    """

    orbital_energies: torch.Tensor
    coefficients: torch.Tensor
    density: torch.Tensor
    energy: float
    n_iterations: int
    density_change: float
    residual: float

    def __post_init__(self) -> None:
        n = self.coefficients.shape[0]
        if self.coefficients.shape != (n, n):
            raise ValidationError(
                "the Hartree-Fock coefficients are square; received shape "
                f"{tuple(self.coefficients.shape)}"
            )
        if self.density.shape != (n, n):
            raise ValidationError(
                "the density matrix has the shape of the coefficient matrix; "
                f"received {tuple(self.density.shape)} against "
                f"{tuple(self.coefficients.shape)}"
            )
        if self.orbital_energies.shape != (n,):
            raise ValidationError(
                "there is one orbital energy per orbital; received "
                f"{tuple(self.orbital_energies.shape)} against {n}"
            )
        if self.n_iterations < 1:
            raise ValidationError(
                f"a converged solve took at least one iteration; received "
                f"{self.n_iterations}"
            )


@dataclass(frozen=True, slots=True)
class MolecularHamiltonian:
    """A molecule's active-space Hamiltonian and the energies around it.

    ``one_body`` and ``two_body`` are the integrals of the *active* space in the
    molecular-orbital basis, with ``two_body[p, q, r, s] = (pq|rs)`` in the
    chemist convention. ``hamiltonian`` is the second-quantised electronic
    Hamiltonian those integrals define, mapped to the Pauli basis by
    :func:`flagquantum.observables.jordan_wigner`; it already carries
    ``nuclear_repulsion + core_energy`` as a constant term, so its ground state is
    an energy rather than an energy difference. ``n_electrons`` and
    ``n_orbitals`` count the active space, which is the space ``hamiltonian`` acts
    on.

    ``hf_energy`` is the restricted Hartree-Fock energy of the whole molecule and
    ``fci_energy`` is the exact ground state of ``hamiltonian`` -- ``None`` when
    the determinant space exceeds :data:`MAX_FCI_DETERMINANTS`, with
    ``fci_determinants`` recording the size it would have had.

    Examples:
        >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
        >>> result = create_molecular_hamiltonian(
        ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        ... )
        >>> result.n_electrons, result.n_orbitals
        (2, 2)
        >>> round(result.hf_energy, 10)
        -1.1166843871
        >>> round(result.fci_energy, 10)
        -1.1372701747
    """

    geometry: MolecularGeometry
    basis: str
    hartree_fock: HartreeFockSolution
    one_body: torch.Tensor
    two_body: torch.Tensor
    hamiltonian: Observable
    n_electrons: int
    n_orbitals: int
    n_frozen_core_orbitals: int
    nuclear_repulsion: float
    core_energy: float
    hf_energy: float
    fci_energy: float | None
    fci_determinants: int

    def __post_init__(self) -> None:
        n = self.n_orbitals
        if self.one_body.shape != (n, n):
            raise ValidationError(
                f"the active one-body integrals are ({n}, {n}); received "
                f"{tuple(self.one_body.shape)}"
            )
        if self.two_body.shape != (n, n, n, n):
            raise ValidationError(
                f"the active two-body integrals are ({n}, {n}, {n}, {n}); received "
                f"{tuple(self.two_body.shape)}"
            )
        if self.n_electrons < 0 or self.n_electrons > 2 * n:
            raise ValidationError(
                f"the active space holds {n} orbitals carrying at most {2 * n} "
                f"electrons; received {self.n_electrons}"
            )
        if self.n_frozen_core_orbitals < 0:
            raise ValidationError(
                "a frozen core is a non-negative number of orbitals; received "
                f"{self.n_frozen_core_orbitals}"
            )

    @property
    def energy_offset(self) -> float:
        """Return the constant the Hamiltonian carries, in hartree."""

        return self.nuclear_repulsion + self.core_energy


def _hartree_fock_energy(
    one_body: torch.Tensor,
    two_body: torch.Tensor,
    density: torch.Tensor,
) -> float:
    """Return the restricted Hartree-Fock energy of ``density``."""

    coulomb = torch.einsum("ls,mnls->mn", density, two_body)
    exchange = torch.einsum("ls,mlns->mn", density, two_body)
    return float(
        torch.einsum("mn,mn->", density, one_body)
        + 0.5 * torch.einsum("mn,mn->", density, coulomb)
        - 0.25 * torch.einsum("mn,mn->", density, exchange)
    )


def restricted_hartree_fock(
    integrals: MolecularIntegrals,
    n_electrons: int,
    *,
    tolerance: float = _SCF_TOLERANCE,
    max_iterations: int = _SCF_MAX_ITERATIONS,
) -> HartreeFockSolution:
    """Return the restricted Hartree-Fock solution of ``integrals``.

    The iteration is the textbook one. The overlap matrix is orthogonalised,
    ``X = U s^-1/2 U^T`` for ``S = U s U^T``; each step diagonalises
    ``X^T F X``, transforms the eigenvectors back to the atomic-orbital basis,
    forms the density ``P = 2 C_occ C_occ^T``, builds the Coulomb and exchange
    matrices from ``P`` and the two-electron integrals, and closes the loop with
    ``F = h + J - K / 2``. The step is undamped: the density is replaced rather
    than mixed with its predecessor, which converges for the eight molecules the
    module documents and two-cycles at a stretched bond, where it raises instead.

    Convergence is decided by the density and not by the energy. The energy is
    stationary at a fixed point, so its change per iteration reaches the
    arithmetic floor while the density is still moving: for lithium hydride the
    energy changes by 1e-13 while ``max |F P S - S P F|`` is still 1e-08. Stopping
    on the energy would return an unconverged density attached to a converged
    number.

    Args:
        integrals: The atomic-orbital integrals of a molecule.
        n_electrons: The number of electrons, even and at most twice the orbital
            count.
        tolerance: The largest density-matrix change that counts as converged.
        max_iterations: The iteration budget. Exhausting it raises rather than
            returning the last iterate.

    Returns:
        The :class:`HartreeFockSolution`, whose ``energy`` already includes the
        nuclear repulsion.

    Raises:
        ValueError: If ``n_electrons`` is not a positive even integer within the
            basis, or if the iteration budget is not positive.
        CapabilityError: If the electron count is odd, if the overlap matrix is
            not positive definite, or if the iteration does not converge within
            the budget.
    """

    n = integrals.n_orbitals
    if not isinstance(n_electrons, int) or isinstance(n_electrons, bool):
        raise ValidationError(
            f"the electron count is an integer; received {type(n_electrons).__name__}"
        )
    if n_electrons <= 0 or n_electrons > 2 * n:
        raise ValidationError(
            f"{n} orbitals carry between one and {2 * n} electrons; received "
            f"{n_electrons}"
        )
    if n_electrons % 2:
        raise CapabilityError(
            "restricted Hartree-Fock pairs every electron, so it needs an even "
            f"electron count; received {n_electrons}, which is an open-shell system "
            "and would need an unrestricted or restricted-open-shell solve that this "
            "module does not carry"
        )
    if max_iterations < 1:
        raise ValidationError(
            f"the iteration budget is at least one; received {max_iterations}"
        )
    overlap = integrals.overlap
    one_body = integrals.core_hamiltonian
    two_body = integrals.two_electron
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
    if bool((eigenvalues <= 0).any()):
        raise CapabilityError(
            "the overlap matrix is not positive definite, so the basis is linearly "
            "dependent and no orthogonalising transformation exists"
        )
    transform = eigenvectors @ torch.diag(eigenvalues.pow(-0.5)) @ eigenvectors.T
    n_occupied = n_electrons // 2
    density = torch.zeros_like(one_body)
    fock = one_body.clone()
    orbital_energies = torch.zeros(n, dtype=one_body.dtype)
    coefficients = torch.eye(n, dtype=one_body.dtype)
    density_change = float("inf")
    for iteration in range(1, max_iterations + 1):
        orbital_energies, eigenvectors = torch.linalg.eigh(
            transform.T @ fock @ transform
        )
        coefficients = transform @ eigenvectors
        occupied = coefficients[:, :n_occupied]
        updated = 2.0 * occupied @ occupied.T
        density_change = float((updated - density).abs().max())
        density = updated
        coulomb = torch.einsum("ls,mnls->mn", density, two_body)
        exchange = torch.einsum("ls,mlns->mn", density, two_body)
        fock = one_body + coulomb - 0.5 * exchange
        if density_change < tolerance:
            break
    else:
        raise CapabilityError(
            f"the Hartree-Fock iteration did not converge in {max_iterations} "
            f"iterations; the largest density change was {density_change:.3e}, above "
            f"the tolerance {tolerance:.3e}"
        )
    return HartreeFockSolution(
        orbital_energies=orbital_energies,
        coefficients=coefficients,
        density=density,
        energy=_hartree_fock_energy(one_body, two_body, density)
        + integrals.nuclear_repulsion,
        n_iterations=iteration,
        density_change=density_change,
        residual=float(
            (fock @ density @ overlap - overlap @ density @ fock).abs().max()
        ),
    )


def molecular_orbital_integrals(
    integrals: MolecularIntegrals,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(one_body, two_body)`` transformed into the given orbitals.

    The two-electron transform is the four-index one applied one index at a time,
    so its cost is the fourth power of the orbital count rather than the eighth.
    Each step carries the coefficient matrix's *column* index into the slot being
    transformed, which is what makes the result the integral over the new orbitals
    rather than over their transpose. The returned arrays are the integrals of the
    orbitals named by the columns of ``coefficients``, in the same chemist
    convention the input uses: ``two_body[p, q, r, s] = (pq|rs)``.
    """

    one_body = coefficients.T @ integrals.core_hamiltonian @ coefficients
    two_body = torch.einsum("mnls,sk->mnlk", integrals.two_electron, coefficients)
    two_body = torch.einsum("mnlk,lr->mnrk", two_body, coefficients)
    two_body = torch.einsum("mnrk,nq->mqrk", two_body, coefficients)
    two_body = torch.einsum("mqrk,mp->pqrk", two_body, coefficients)
    return one_body, two_body.contiguous()


def fold_frozen_core(
    one_body: torch.Tensor,
    two_body: torch.Tensor,
    n_frozen_core: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fold the lowest ``n_frozen_core`` orbitals into the active space.

    Holding those orbitals doubly occupied turns their one-electron energy into a
    constant, their Coulomb and exchange interaction with the active orbitals into
    a shift of the active one-body integrals, and their mutual interaction into a
    second constant. Nothing is dropped, so the reduced Hamiltonian's eigenvalues
    are the full Hamiltonian's restricted to determinants with those orbitals
    occupied.
    """

    n = one_body.shape[0]
    active = tuple(range(n_frozen_core, n))
    core_energy = 0.0
    for i in range(n_frozen_core):
        core_energy += 2.0 * float(one_body[i, i])
        for j in range(n_frozen_core):
            core_energy += 2.0 * float(two_body[i, i, j, j]) - float(
                two_body[i, j, j, i]
            )
    effective = one_body.clone()
    for p in active:
        for q in active:
            shift = 0.0
            for i in range(n_frozen_core):
                shift += 2.0 * float(two_body[p, q, i, i]) - float(two_body[p, i, i, q])
            effective[p, q] = one_body[p, q] + shift
    selected = torch.tensor(active, dtype=torch.long)
    active_one_body = effective.index_select(0, selected).index_select(1, selected)
    active_two_body = (
        two_body.index_select(0, selected)
        .index_select(1, selected)
        .index_select(2, selected)
        .index_select(3, selected)
    )
    return active_one_body.contiguous(), active_two_body.contiguous(), core_energy


def second_quantized_hamiltonian(
    one_body: torch.Tensor,
    two_body: torch.Tensor,
    constant: float = 0.0,
) -> FermionOperator:
    """Return the electronic Hamiltonian of the given integrals.

    Spin orbitals are paired in the order
    :func:`flagquantum.algorithms.chemistry.uccsd_excitations` assumes: ``2s`` is
    the alpha spin orbital of spatial orbital ``s`` and ``2s + 1`` its beta
    partner. The one-body part couples equal spins only, the two-electron part
    enumerates the four surviving spin patterns, and ``constant`` enters as a
    degree-zero term so the operator's eigenvalues are energies. The result is
    exactly Hermitian, which is the property
    :func:`flagquantum.observables.jordan_wigner` requires of it.
    """

    n = one_body.shape[0]
    creations = [create(index) for index in range(2 * n)]
    annihilations = [annihilate(index) for index in range(2 * n)]
    terms: list[FermionTerm] = []
    if constant != 0.0:
        terms.append(FermionTerm(constant))
    for left in range(2 * n):
        for right in range(2 * n):
            if left % 2 != right % 2:
                continue
            value = float(one_body[left // 2, right // 2])
            if value == 0.0:
                continue
            terms.extend((value * (creations[left] * annihilations[right])).terms)
    for p in range(n):
        for r in range(n):
            material = two_body[p, r]
            for q in range(n):
                for s in range(n):
                    value = 0.5 * float(material[q, s])
                    if value == 0.0:
                        continue
                    for pattern in _SPIN_PATTERNS:
                        monomial = (
                            creations[2 * p + pattern[0]]
                            * creations[2 * q + pattern[1]]
                        )
                        monomial = (
                            monomial
                            * annihilations[2 * s + pattern[3]]
                            * annihilations[2 * r + pattern[2]]
                        )
                        terms.extend((value * monomial).terms)
    assembled = FermionOperator(tuple(terms))
    return 0.5 * (assembled + assembled.dagger())


def _determinant_masks(n_orbitals: int, n_electrons: int) -> list[int]:
    """Return the bitmasks of the ``S_z = 0`` determinants, lowest first."""

    n_alpha = n_electrons // 2
    alpha = [
        sum(1 << (2 * orbital) for orbital in combination)
        for combination in itertools.combinations(range(n_orbitals), n_alpha)
    ]
    beta = [
        sum(1 << (2 * orbital + 1) for orbital in combination)
        for combination in itertools.combinations(range(n_orbitals), n_alpha)
    ]
    return [a | b for a in alpha for b in beta]


def _sector_dimension(n_orbitals: int, n_electrons: int) -> int:
    """Return the number of ``S_z = 0`` determinants without enumerating them."""

    n_alpha = n_electrons // 2
    return len(tuple(itertools.combinations(range(n_orbitals), n_alpha))) ** 2


def exact_ground_state_energy(
    operator: FermionOperator,
    *,
    n_electrons: int,
    n_orbitals: int,
) -> float:
    """Return the lowest eigenvalue of ``operator`` in the ``S_z = 0`` sector.

    The sector fixes half the electrons in the alpha spin orbitals and half in the
    beta ones, which for a closed-shell system contains the ground state and holds
    ``C(n_orbitals, n_electrons / 2) ** 2`` determinants rather than the
    ``C(2 n_orbitals, n_electrons)`` of the whole fixed-particle-number space. The
    matrix is built in that basis by applying each monomial to each determinant,
    with the sign taken from the parity of the occupied modes below the one being
    moved.

    Args:
        operator: The Hamiltonian, including any constant term it carries.
        n_electrons: The number of electrons to hold fixed, even.
        n_orbitals: The number of spatial orbitals the operator acts on.

    Returns:
        The lowest eigenvalue as a float, with the operator's constant included.

    Raises:
        ValueError: If ``n_electrons`` is not an even integer in ``[2, 2 n_orbitals]``.
        CapabilityError: If the sector holds more than
            :data:`MAX_FCI_DETERMINANTS` determinants.
    """

    if (
        not isinstance(n_electrons, int)
        or isinstance(n_electrons, bool)
        or n_electrons <= 0
        or n_electrons > 2 * n_orbitals
        or n_electrons % 2
    ):
        raise ValidationError(
            "the S_z = 0 sector needs an even electron count between two and "
            f"{2 * n_orbitals}; received {n_electrons}"
        )
    dimension = _sector_dimension(n_orbitals, n_electrons)
    if dimension > MAX_FCI_DETERMINANTS:
        raise CapabilityError(
            f"the S_z = 0 sector of {n_electrons} electrons in {n_orbitals} orbitals "
            f"holds {dimension} determinants, above the {MAX_FCI_DETERMINANTS} this "
            "module diagonalises exactly"
        )
    basis = _determinant_masks(n_orbitals, n_electrons)
    positions = {mask: index for index, mask in enumerate(basis)}
    matrix = torch.zeros((dimension, dimension), dtype=torch.float64)
    for term in operator.terms:
        coefficient = float(term.coefficient.real)
        if coefficient == 0.0:
            continue
        sequence: list[tuple[int, bool]] = [(mode, True) for mode in term.creations]
        sequence.extend((mode, False) for mode in term.annihilations)
        for column, mask in enumerate(basis):
            occupied = mask
            sign = 1.0
            for mode, creating in reversed(sequence):
                bit = 1 << mode
                if creating:
                    if occupied & bit:
                        sign = 0.0
                        break
                    if bin(occupied & (bit - 1)).count("1") & 1:
                        sign = -sign
                    occupied |= bit
                else:
                    if not occupied & bit:
                        sign = 0.0
                        break
                    if bin(occupied & (bit - 1)).count("1") & 1:
                        sign = -sign
                    occupied &= ~bit
            if sign == 0.0:
                continue
            row = positions.get(occupied)
            if row is not None:
                matrix[row, column] += coefficient * sign
    return float(torch.linalg.eigvalsh(matrix)[0])


def create_molecular_hamiltonian(
    geometry: Iterable[tuple[str, Sequence[float]]] | MolecularGeometry,
    *,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    charge: int = 0,
    n_active_electrons: int | None = None,
    n_active_orbitals: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> MolecularHamiltonian:
    """Return the qubit Hamiltonian of ``geometry`` in ``basis``.

    The call evaluates the integrals, solves restricted Hartree-Fock, transforms
    both integral arrays into the canonical molecular orbitals, folds away the
    frozen core an active space asks for, assembles the second-quantised
    Hamiltonian of the active integrals, and maps it onto Pauli operators with
    :func:`flagquantum.observables.jordan_wigner`.

    A ``driver`` argument is deliberately absent. CUDA-Q's C++ entry point accepts
    one, resolves the driver out of a registry keyed on the literal ``"pyscf"``, and
    ignores the value it was given; accepting a string here that could not change
    the answer would be a second, silent source of truth about which integrals were
    computed.

    Args:
        geometry: ``(element, (x, y, z))`` pairs in angstroms, or a
            :class:`flagquantum.algorithms.chemistry_integrals.MolecularGeometry`.
        basis: The basis set. STO-3G is the only one available.
        multiplicity: The spin multiplicity. Only one is supported, because the
            solve is restricted and closed-shell.
        charge: The total charge. The electron count is the nuclear charge minus
            this, and it has to come out positive and even.
        n_active_electrons: The number of electrons to keep active. The difference
            from the total, halved, is the number of frozen-core orbitals.
        n_active_orbitals: The number of orbitals to keep active, counted from the
            first orbital above the frozen core. ``None`` keeps all of them.
        dtype: The dtype of the returned integral arrays. They are always
            accumulated in float64 and cast at the end, and the Hamiltonian is
            built in float64 whatever this is.

    Returns:
        The :class:`MolecularHamiltonian`.

    Raises:
        ValueError: If the geometry, the charge, the electron count or the
            active-space counts are inconsistent.
        CapabilityError: If the multiplicity is not one, if an element is outside
            the STO-3G table, if the basis is not STO-3G, or if the Hartree-Fock
            iteration does not converge.

    Examples:
        >>> from flagquantum.algorithms.molecular import create_molecular_hamiltonian
        >>> result = create_molecular_hamiltonian(
        ...     [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        ... )
        >>> result.n_orbitals, len(result.hamiltonian.terms)
        (2, 27)
    """

    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
        raise ValidationError(
            f"the multiplicity is an integer; received {type(multiplicity).__name__}"
        )
    if multiplicity != 1:
        raise CapabilityError(
            "the Hartree-Fock solve is restricted and closed-shell, so it needs a "
            f"singlet; received multiplicity {multiplicity}, which needs an "
            "unrestricted or restricted-open-shell solve that this module does not "
            "carry"
        )
    if not isinstance(charge, int) or isinstance(charge, bool):
        raise ValidationError(
            f"the charge is an integer; received {type(charge).__name__}"
        )
    molecule = (
        geometry
        if isinstance(geometry, MolecularGeometry)
        else MolecularGeometry.from_angstrom(geometry)
    )
    n_electrons = molecule.n_electrons - charge
    if n_electrons <= 0:
        raise ValidationError(
            f"a charge of {charge} leaves {n_electrons} electrons in "
            f"{' + '.join(molecule.symbols)}; a molecule needs at least one electron"
        )
    if n_active_orbitals is not None and (
        not isinstance(n_active_orbitals, int) or isinstance(n_active_orbitals, bool)
    ):
        raise ValidationError(
            "the active orbital count is an integer or None; received "
            f"{type(n_active_orbitals).__name__}"
        )
    if n_active_electrons is None:
        n_frozen_core = 0
    else:
        if not isinstance(n_active_electrons, int) or isinstance(
            n_active_electrons, bool
        ):
            raise ValidationError(
                "the active electron count is an integer or None; received "
                f"{type(n_active_electrons).__name__}"
            )
        if n_active_electrons <= 0 or n_active_electrons > n_electrons:
            raise ValidationError(
                f"the active space keeps between one and {n_electrons} electrons; "
                f"received {n_active_electrons}"
            )
        if (n_electrons - n_active_electrons) % 2:
            raise ValidationError(
                "the frozen core holds whole electron pairs, so the active electron "
                f"count has the parity of the total {n_electrons}; received "
                f"{n_active_electrons}"
            )
        n_frozen_core = (n_electrons - n_active_electrons) // 2
    integrals = molecular_integrals(molecule, basis=basis, dtype=torch.float64)
    solve = restricted_hartree_fock(integrals, n_electrons)
    one_body, two_body = molecular_orbital_integrals(integrals, solve.coefficients)
    n_above_core = integrals.n_orbitals - n_frozen_core
    if n_active_orbitals is None:
        n_active = n_above_core
    else:
        if n_active_orbitals < 1 or n_active_orbitals > n_above_core:
            raise ValidationError(
                f"an active space keeps between one and the {n_above_core} orbitals "
                f"above a frozen core of {n_frozen_core}; received {n_active_orbitals}"
            )
        n_active = n_active_orbitals
    one_body, two_body, core_energy = fold_frozen_core(
        one_body, two_body, n_frozen_core
    )
    if n_active < n_above_core:
        selected = torch.arange(n_active, dtype=torch.long)
        one_body = one_body.index_select(0, selected).index_select(1, selected)
        two_body = (
            two_body.index_select(0, selected)
            .index_select(1, selected)
            .index_select(2, selected)
            .index_select(3, selected)
        )
        one_body = one_body.contiguous()
        two_body = two_body.contiguous()
    n_active_electrons = n_electrons - 2 * n_frozen_core
    if n_active_electrons > 2 * n_active:
        raise ValidationError(
            f"{n_active} active orbitals carry at most {2 * n_active} electrons; "
            f"{n_active_electrons} are left after freezing {n_frozen_core} orbitals"
        )
    operator = second_quantized_hamiltonian(
        one_body,
        two_body,
        integrals.nuclear_repulsion + core_energy,
    )
    determinants = _sector_dimension(n_active, n_active_electrons)
    fci_energy: float | None = None
    if determinants <= MAX_FCI_DETERMINANTS:
        fci_energy = exact_ground_state_energy(
            operator, n_electrons=n_active_electrons, n_orbitals=n_active
        )
    return MolecularHamiltonian(
        geometry=molecule,
        basis=basis,
        hartree_fock=solve,
        one_body=one_body.to(dtype=dtype),
        two_body=two_body.to(dtype=dtype),
        hamiltonian=jordan_wigner(operator, n_modes=2 * n_active),
        n_electrons=n_active_electrons,
        n_orbitals=n_active,
        n_frozen_core_orbitals=n_frozen_core,
        nuclear_repulsion=integrals.nuclear_repulsion,
        core_energy=core_energy,
        hf_energy=solve.energy,
        fci_energy=fci_energy,
        fci_determinants=determinants,
    )
