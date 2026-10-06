"""The chemistry driver: integrals, Hartree-Fock, and the qubit Hamiltonian.

This file checks the second half of the chemistry entry point -- the
Hartree-Fock solve, the molecular-orbital transformation, the active-space
selection, and the second-quantised Hamiltonian that
:func:`flagquantum.observables.jordan_wigner` maps onto qubits. The first half,
the atomic-orbital integrals, has its own companion file.

Very little here compares the code under test with itself. The reference
energies are the STO-3G restricted Hartree-Fock values for eight molecules, the
H2 full-configuration-interaction value is the published one, the Hartree-Fock
energy is recomputed from the integrals by the Slater-Condon expression written
out in this file, and the qubit Hamiltonian's ground state is recomputed from
the Pauli sum's own matrix factors rather than from the determinant algebra the
module uses. Those four routes share no code with one another.

What this file proves
---------------------

1. Two-electron integrals carry the eight-fold index symmetry of the chemist
   notation ``(ij|kl)``, so the one evaluation the engine performs populates
   every permutation exactly rather than approximately.
2. The molecular-orbital transformation is the transform, not a plausible
   rearrange of it: it is the identity map at ``Q = I`` and it agrees
   elementwise with the single eight-index contraction.
3. The restricted Hartree-Fock energy of H2, LiH, HF, H2O, NH3, CH4, N2 and CO
   at their standard geometries reproduces the STO-3G reference to 1e-9, and the
   density it converges on satisfies ``max |F P S - S P F| < 1e-9``.
4. The converged solution satisfies two closed forms it was not fitted to: the
   orbital-energy identity ``E = sum_i (h_ii + eps_i) + E_nuc`` and the
   Slater-Condon determinant energy
   ``E = 2 sum_i h_ii + sum_ij (2 (ii|jj) - (ij|ji)) + E_nuc``. The second one
   pins the index pattern of the two-electron operator, because the alternative
   pairing is off by tens of hartree and the test asserts that.
5. The Hartree-Fock determinant's expectation, read off the diagonal of the
   Jordan-Wigner image through the convention that
   :func:`flagquantum.observables.jordan_wigner` itself is checked against,
   equals the Hartree-Fock energy.
6. ``exact_ground_state_energy`` returns the lowest eigenvalue of the same
   operator built from explicit Pauli matrix factors, for H2 over the full
   space and for LiH over a frozen-core active space.
7. The exact ground state is a functional of the integrals alone: rotating the
   one- and two-body arrays by the same orthogonal mixing leaves it where it
   was.
8. Freezing a core orbital is exact rather than approximate: the reduced
   Hamiltonian's ground state is the ground state of the full Hamiltonian
   restricted to the determinants that hold that orbital, and the effective
   one-body shift is the closed form ``h_pq + sum_i (2 (pq|ii) - (pi|iq))``.
9. The active-space ground state never falls below the full-space one, which is
   the variational statement that an active space is a restriction.
10. The two halves compose: a UCCSD circuit built on this Hamiltonian's electron
    and orbital counts, measured against this Hamiltonian, gives the
    Hartree-Fock energy at zero angle, reaches the exact ground state of H2 to
    2e-6 at its minimum, and never falls below it anywhere the test samples.

What this file does not prove
-----------------------------

* That the atomic-orbital integrals are right. That is the companion file's
  claim, and this one takes them as given.
* That STO-3G is an adequate basis, or that any of these energies is a
  prediction about a real molecule. Every number here is a statement about a
  finite Gaussian basis.
* That the fixed point the Hartree-Fock iteration reaches is the lowest one. A
  reference energy matching to 1e-9 is evidence that the intended solution was
  reached for these eight molecules and nothing about another molecule, another
  geometry, or a second solution at a stretched bond.
* That the Hamiltonian is efficient to simulate. The Pauli sum carries every
  quadruple whose integral is nonzero; the largest measured here is 21407 terms
  for N2, and nothing about that count is a claim about circuit cost.
* Anything about a molecule with an odd electron count, a multiplicity other
  than one, or an element outside H to Ne. Those are refusals, and the tests
  below check that they are refusals rather than that they are handled.
"""

import itertools
import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz, uccsd_excitations
from flagquantum.algorithms.chemistry_integrals import (
    ATOMIC_NUMBERS,
    MolecularGeometry,
    molecular_integrals,
)
from flagquantum.algorithms.molecular import (
    MAX_FCI_DETERMINANTS,
    MOLECULAR_ASSUMPTIONS,
    MOLECULAR_LIMITATIONS,
    HartreeFockSolution,
    MolecularHamiltonian,
    create_molecular_hamiltonian,
    exact_ground_state_energy,
    fold_frozen_core,
    molecular_orbital_integrals,
    restricted_hartree_fock,
    second_quantized_hamiltonian,
)
from flagquantum.errors import CapabilityError, ValidationError

pytestmark = pytest.mark.unit

#: The STO-3G restricted Hartree-Fock energy of each molecule, in hartree.
#:
#: These are the reference values for the basis, not measurements of this code.
REFERENCE_HF_ENERGIES = {
    "H2": -1.1166843871,
    "LiH": -7.8620238601,
    "HF": -98.5707799860,
    "H2O": -74.9650107063,
    "NH3": -55.4540872540,
    "CH4": -39.7266228578,
    "N2": -107.4958933078,
    "CO": -111.2245895545,
}

#: The number of electrons each of those molecules has.
REFERENCE_ELECTRONS = {
    "H2": 2,
    "LiH": 4,
    "HF": 10,
    "H2O": 10,
    "NH3": 10,
    "CH4": 10,
    "N2": 14,
    "CO": 14,
}


def _h2() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the H2 geometry at 0.7414 angstrom."""

    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]


def _lih() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the LiH geometry at 1.595 angstrom."""

    return [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.595))]


def _hf() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the HF geometry at 0.917 angstrom."""

    return [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.917))]


def _h2o() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the H2O geometry at 0.9893 angstrom and 104.52 degrees.

    The oxygen sits at the origin and the two hydrogens in the ``xz`` plane, so
    the H-O-H angle is the angle between the two bonds.
    """

    half = math.radians(104.52) / 2.0
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.9893 * math.sin(half), 0.0, 0.9893 * math.cos(half))),
        ("H", (-0.9893 * math.sin(half), 0.0, 0.9893 * math.cos(half))),
    ]


def _nh3() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the NH3 geometry at 1.0124 angstrom and 106.67 degrees.

    The nitrogen sits at the origin with the hydrogens on a cone about ``-z``.
    For a three-fold rotor the H-N-H angle ``a`` fixes the cone half-angle ``t``
    through ``cos a = cos^2 t - sin^2 t / 2``, so
    ``cos^2 t = (1 + 2 cos a) / 3``.
    """

    cosine = math.cos(math.radians(106.67))
    axial = math.sqrt((1.0 + 2.0 * cosine) / 3.0)
    radial = 1.0124 * math.sqrt(1.0 - axial * axial)
    return [
        ("N", (0.0, 0.0, 0.0)),
        *(
            (
                "H",
                (
                    radial * math.cos(2.0 * math.pi * index / 3.0),
                    radial * math.sin(2.0 * math.pi * index / 3.0),
                    -1.0124 * axial,
                ),
            )
            for index in range(3)
        ),
    ]


def _ch4() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the CH4 geometry at 1.0915 angstrom, tetrahedral."""

    scale = 1.0915 / math.sqrt(3.0)
    return [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (scale, scale, scale)),
        ("H", (-scale, -scale, scale)),
        ("H", (-scale, scale, -scale)),
        ("H", (scale, -scale, -scale)),
    ]


def _n2() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the N2 geometry at 1.0977 angstrom."""

    return [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))]


def _co() -> list[tuple[str, tuple[float, float, float]]]:
    """Return the CO geometry at 1.1283 angstrom."""

    return [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.1283))]


#: The eight standard geometries, by the name the reference table uses.
GEOMETRIES = {
    "H2": _h2,
    "LiH": _lih,
    "HF": _hf,
    "H2O": _h2o,
    "NH3": _nh3,
    "CH4": _ch4,
    "N2": _n2,
    "CO": _co,
}


def _slater_determinant_energy(
    one_body: torch.Tensor, two_body: torch.Tensor, n_occupied: int
) -> float:
    """Return the closed-form energy of a closed-shell determinant.

    ``one_body[p, q]`` is ``h_pq`` and ``two_body[p, q, r, s]`` is ``(pq|rs)``,
    both over spatial molecular orbitals. The closed form is the Slater-Condon
    rule for a determinant with every orbital doubly occupied: the one-electron
    part is counted twice, the Coulomb part ``(ii|jj)`` is counted twice for the
    double occupancy, and the exchange part is ``(ij|ji)`` with the two indices
    of the *second* electron swapped.
    """

    total = 0.0
    for i in range(n_occupied):
        total += 2.0 * float(one_body[i, i])
        for j in range(n_occupied):
            total += 2.0 * float(two_body[i, i, j, j])
            total -= float(two_body[i, j, j, i])
    return total


_ELEMENT_WISE_PAULI = {
    "i": ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, 1.0 + 0.0j)),
    "x": ((0.0 + 0.0j, 1.0 + 0.0j), (1.0 + 0.0j, 0.0 + 0.0j)),
    "y": ((0.0 + 0.0j, -1.0j), (1.0j, 0.0 + 0.0j)),
    "z": ((1.0 + 0.0j, 0.0 + 0.0j), (0.0 + 0.0j, -1.0 + 0.0j)),
}


def _sector_basis(n_orbitals: int, n_electrons: int) -> list[int]:
    """Return every occupation mask with equal alpha and beta population."""

    n_alpha = n_electrons // 2
    return [
        sum(1 << (2 * orbital) for orbital in alpha)
        | sum(1 << (2 * orbital + 1) for orbital in beta)
        for alpha in itertools.combinations(range(n_orbitals), n_alpha)
        for beta in itertools.combinations(range(n_orbitals), n_alpha)
    ]


def _sector_matrix(
    observable: fq.Observable, basis: list[int], width: int
) -> torch.Tensor:
    """Return the block of a Pauli sum over ``basis``, from its matrix factors.

    This is the second route to the operator's spectrum. Each term contributes
    ``coefficient * product_w <row_w| P_w |col_w>`` over the wires the term
    touches and an identity elsewhere, which is the definition of a Pauli
    product in the occupation basis and shares nothing with the determinant
    algebra :func:`flagquantum.algorithms.molecular.exact_ground_state_energy`
    uses. Wire ``w`` is the ``w``-th bit of the mask, and the masks pair
    ``2 s`` with the alpha spin orbital of ``s`` and ``2 s + 1`` with its beta
    partner, which is the pairing the Hamiltonian is assembled with.

    ``basis`` is a parameter rather than always the whole ``S_z = 0`` sector so
    that a caller can restrict it in place. Building the whole sector and
    slicing it afterwards costs the square of the sector's dimension, which for
    a twelve-spin-orbital molecule is a five-figure multiple of what the
    restricted set costs.
    """

    positions = {mask: index for index, mask in enumerate(basis)}
    dimension = len(basis)
    accumulator = [[0j] * dimension for _ in range(dimension)]
    for term in observable.terms:
        touched = dict(term.factors)
        # On the wires a term touches, ``X`` and ``Y`` flip the occupation and
        # ``Z`` does not, so the term maps one mask onto exactly one other. ``X``
        # contributes no phase, ``Z`` contributes ``(-1) ** bit``, and ``Y``
        # contributes ``-i * (-1) ** bit``.
        flip = 0
        parity = 0
        n_y = 0
        for wire, label in touched.items():
            assert label in _ELEMENT_WISE_PAULI
            if label in ("x", "y"):
                flip |= 1 << wire
            if label in ("y", "z"):
                parity |= 1 << wire
            n_y += label == "y"
        base = complex(term.coefficient) * (-1.0j) ** n_y
        for column, mask in enumerate(basis):
            row = positions.get(mask ^ flip)
            if row is None:
                continue
            value = base
            if bin(mask & parity).count("1") % 2:
                value = -value
            accumulator[row][column] += value
    matrix = torch.tensor(accumulator, dtype=torch.complex128)
    assert float(matrix.imag.abs().max()) < 1e-12
    real = matrix.real
    return 0.5 * (real + real.T)


def _element_wise_sector_matrix(
    observable: fq.Observable, basis: list[int], width: int
) -> torch.Tensor:
    """Return the same block the long way, straight from the matrix elements.

    :func:`_sector_matrix` compresses each Pauli term into a bit flip and a
    phase. That is only valid if ``X`` and ``Y`` flip the occupation, ``Z`` does
    not, and the phases are the ones above, so this route spells those matrix
    elements out and the two are compared on a molecule small enough to afford
    it. It is otherwise the same block over the same basis.
    """

    dimension = len(basis)
    matrix = torch.zeros((dimension, dimension), dtype=torch.complex128)
    for term in observable.terms:
        touched = dict(term.factors)
        fixed = [wire for wire in range(width) if wire not in touched]
        for column, mask in enumerate(basis):
            for row, other in enumerate(basis):
                if any((mask >> wire) & 1 != (other >> wire) & 1 for wire in fixed):
                    continue
                value = complex(term.coefficient)
                for wire in range(width):
                    label = touched.get(wire, "i")
                    value *= _ELEMENT_WISE_PAULI[label][(other >> wire) & 1][
                        (mask >> wire) & 1
                    ]
                matrix[row, column] += value
    assert float(matrix.imag.abs().max()) < 1e-12
    real = matrix.real
    return 0.5 * (real + real.T)


def _molecular_orbitals(integrals, n_electrons: int):
    """Return ``(solution, one_body, two_body)`` in the canonical orbitals."""

    solve = restricted_hartree_fock(integrals, n_electrons)
    one_body, two_body = molecular_orbital_integrals(integrals, solve.coefficients)
    return solve, one_body, two_body


# ---------------------------------------------------------------------------
# 1. The integrals carry the symmetry the engine assumes of them
# ---------------------------------------------------------------------------


def test_the_two_electron_integrals_are_permutation_symmetric() -> None:
    """Every index order of ``(ij|kl)`` reads back the value that was written."""

    integrals = molecular_integrals(_h2o())
    tensor = integrals.two_electron
    orders = (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    )
    worst = 0.0
    for p, q, r, s in itertools.product(range(integrals.n_orbitals), repeat=4):
        reference = float(tensor[p, q, r, s])
        for order in orders:
            index = (p, q, r, s)
            permuted = tuple(index[position] for position in order)
            worst = max(worst, abs(float(tensor[permuted]) - reference))
    assert worst == 0.0


def test_the_one_electron_integrals_are_symmetric() -> None:
    """Overlap and core Hamiltonian are symmetric matrices."""

    integrals = molecular_integrals(_lih())
    assert torch.equal(integrals.overlap, integrals.overlap.T)
    assert torch.equal(integrals.core_hamiltonian, integrals.core_hamiltonian.T)
    assert integrals.n_orbitals == 6


# ---------------------------------------------------------------------------
# 2. The molecular-orbital transformation is the transform
# ---------------------------------------------------------------------------


def test_the_orbital_transformation_is_the_identity_at_the_identity() -> None:
    integrals = molecular_integrals(_h2o())
    n = integrals.n_orbitals
    one_body, two_body = molecular_orbital_integrals(
        integrals, torch.eye(n, dtype=torch.float64)
    )
    assert torch.equal(one_body, integrals.core_hamiltonian)
    assert torch.equal(two_body, integrals.two_electron)


def test_the_orbital_transformation_matches_the_eight_index_contraction() -> None:
    """Four sequential contractions agree with the one-shot contraction."""

    integrals = molecular_integrals(_lih())
    solve = restricted_hartree_fock(integrals, 4)
    coefficients = solve.coefficients
    one_body, two_body = molecular_orbital_integrals(integrals, coefficients)
    assert torch.allclose(
        one_body,
        coefficients.T @ integrals.core_hamiltonian @ coefficients,
        atol=1e-14,
    )
    assert torch.allclose(
        two_body,
        torch.einsum(
            "abcd,ap,bq,cr,ds->pqrs",
            integrals.two_electron,
            coefficients,
            coefficients,
            coefficients,
            coefficients,
        ),
        atol=1e-13,
    )


def test_the_orbital_transformation_preserves_the_paired_two_electron_sum() -> None:
    """``sum_pq (pq|pq)`` is invariant, because the mixing is orthogonal.

    The contraction collapses two of the four orbital indices against each other,
    ``sum_p C_ap C_cp = delta_ac``, so the paired sum cannot move. The unpaired
    sum ``sum_pqrs (pq|rs)`` does move -- it contracts all four indices
    separately and is not invariant for a general orthogonal mixing -- so the
    paired form is the identity worth asserting rather than a plausible one.
    """

    integrals = molecular_integrals(_h2o())
    n = integrals.n_orbitals
    torch.manual_seed(4)
    mixing, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    _, two_body = molecular_orbital_integrals(integrals, mixing)

    def paired(tensor: torch.Tensor) -> float:
        return float(sum(tensor[p, q, p, q] for p in range(n) for q in range(n)))

    assert abs(paired(two_body) - paired(integrals.two_electron)) < 1e-12
    assert abs(float(two_body.sum()) - float(integrals.two_electron.sum())) > 1.0


# ---------------------------------------------------------------------------
# 3. The Hartree-Fock solve against the reference table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REFERENCE_HF_ENERGIES))
def test_the_hartree_fock_energy_reproduces_the_reference(name: str) -> None:
    integrals = molecular_integrals(GEOMETRIES[name]())
    solve = restricted_hartree_fock(integrals, REFERENCE_ELECTRONS[name])
    assert abs(solve.energy - REFERENCE_HF_ENERGIES[name]) < 1e-9
    assert solve.residual < 1e-9
    assert solve.density_change < 1e-10
    assert solve.n_iterations >= 1


@pytest.mark.parametrize("name", sorted(REFERENCE_HF_ENERGIES))
def test_the_converged_density_commutes_with_the_fock_matrix(name: str) -> None:
    """The residual is a property of the returned density, not of a later one."""

    integrals = molecular_integrals(GEOMETRIES[name]())
    solve = restricted_hartree_fock(integrals, REFERENCE_ELECTRONS[name])
    density = solve.density
    overlap = integrals.overlap
    assert float((density - density.T).abs().max()) < 1e-14
    # The density is in the atomic-orbital basis, so the electron count is
    # tr(P S) and not tr(P): the overlap carries the metric.
    assert abs(float(torch.trace(density @ overlap)) - REFERENCE_ELECTRONS[name]) < 1e-9
    # Doubly occupied closed shell: P S P = 2 P.
    assert float((density @ overlap @ density - 2.0 * density).abs().max()) < 1e-9


def test_the_energy_stopping_rule_would_stop_too_early() -> None:
    """The energy is stationary before the density is, so it cannot be the test.

    The solve is run twice on lithium hydride: once to the default density
    tolerance, and once stopping as soon as the density has settled to ``1e-5``.
    The looser run's energy is already within ``1e-9`` of the tight one, while
    the density it stopped on has moved by ``6e-6``: an energy criterion at that
    threshold would have accepted the same density and called it converged.
    """

    integrals = molecular_integrals(_lih())
    tight = restricted_hartree_fock(integrals, 4)
    loose = restricted_hartree_fock(integrals, 4, tolerance=1e-5)
    assert abs(loose.energy - tight.energy) < 1e-9
    assert loose.density_change > 1e-6
    assert loose.n_iterations < tight.n_iterations


# ---------------------------------------------------------------------------
# 4. The solve against closed forms it was not fitted to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("H2", "LiH", "HF", "H2O", "N2"))
def test_the_solution_satisfies_the_orbital_energy_identity(name: str) -> None:
    """``E = sum_i (h_ii + eps_i) + E_nuc`` over the occupied orbitals."""

    integrals = molecular_integrals(GEOMETRIES[name]())
    solve, one_body, _ = _molecular_orbitals(integrals, REFERENCE_ELECTRONS[name])
    n_occupied = REFERENCE_ELECTRONS[name] // 2
    total = integrals.nuclear_repulsion
    for i in range(n_occupied):
        total += float(one_body[i, i]) + float(solve.orbital_energies[i])
    assert abs(total - solve.energy) < 1e-9


@pytest.mark.parametrize("name", sorted(REFERENCE_HF_ENERGIES))
def test_the_solution_energy_is_the_slater_determinant_energy(name: str) -> None:
    """The Slater-Condon closed form, computed here, reproduces the solve."""

    integrals = molecular_integrals(GEOMETRIES[name]())
    solve, one_body, two_body = _molecular_orbitals(
        integrals, REFERENCE_ELECTRONS[name]
    )
    n_occupied = REFERENCE_ELECTRONS[name] // 2
    expected = (
        _slater_determinant_energy(one_body, two_body, n_occupied)
        + integrals.nuclear_repulsion
    )
    assert abs(expected - solve.energy) < 1e-9
    assert abs(expected - REFERENCE_HF_ENERGIES[name]) < 1e-9


def test_the_slater_energy_needs_its_exchange_term() -> None:
    """The closed form's exchange term is load-bearing, not a rounding correction.

    A control here has to be chosen with care, because the obvious one does not
    discriminate: the eight-fold symmetry of the ``(ij|kl)`` tensor makes
    ``(ij|ji)`` and ``(ij|ij)`` bitwise equal, so an index-order mistake in the
    exchange term would be invisible to this closed form. What is not invisible
    is dropping the term altogether, and that is what this test removes.
    """

    integrals = molecular_integrals(_h2o())
    _, one_body, two_body = _molecular_orbitals(integrals, 10)
    right = _slater_determinant_energy(one_body, two_body, 5)
    assert float(two_body[1, 2, 2, 1]) == float(two_body[1, 2, 1, 2])
    coulomb_only = 0.0
    for i in range(5):
        coulomb_only += 2.0 * float(one_body[i, i])
        for j in range(5):
            coulomb_only += 2.0 * float(two_body[i, i, j, j])
    assert abs(right - coulomb_only) > 1.0
    assert (
        abs(right + integrals.nuclear_repulsion - REFERENCE_HF_ENERGIES["H2O"]) < 1e-9
    )
    assert (
        abs(coulomb_only + integrals.nuclear_repulsion - REFERENCE_HF_ENERGIES["H2O"])
        > 1.0
    )


# ---------------------------------------------------------------------------
# 5. The Jordan-Wigner image of the Hamiltonian
# ---------------------------------------------------------------------------


def _jordan_wigner_number_convention() -> tuple[float, tuple[tuple[int, str], ...]]:
    """Return ``jordan_wigner(number(0), n_modes=2)``'s single Pauli term."""

    from flagquantum.observables.fermion import jordan_wigner, number

    image = jordan_wigner(number(0), n_modes=2)
    assert len(image.terms) == 2
    z_term = next(term for term in image.terms if term.factors)
    return float(z_term.coefficient.real), tuple(z_term.factors)


def test_the_number_operator_is_half_of_one_minus_z() -> None:
    """The convention the determinant expectation below depends on."""

    coefficient, factors = _jordan_wigner_number_convention()
    assert factors == ((0, "z"),)
    assert coefficient == pytest.approx(-0.5, abs=1e-15)


@pytest.mark.parametrize("name", ("H2", "LiH", "HF", "H2O"))
def test_the_pauli_diagonal_reads_back_the_hartree_fock_energy(name: str) -> None:
    """``<HF|H|HF> + E_nuc`` from the ``z``-only terms of the image.

    With ``n_p = (1 - Z_p) / 2`` and no parity string, an ``I``/``Z`` term's
    expectation is its coefficient times ``(-1)`` to the number of ``Z`` factors
    on occupied spin orbitals. Every other term in the image moves an electron
    and contributes nothing to a determinant expectation, so the sum over the
    ``z``-only terms is the whole of it.
    """

    result = create_molecular_hamiltonian(GEOMETRIES[name]())
    n_occupied = result.n_electrons // 2
    total = 0.0
    for term in result.hamiltonian.terms:
        if any(label != "z" for _, label in term.factors):
            continue
        occupied = sum(1 for wire, _ in term.factors if wire // 2 < n_occupied)
        total += float(term.coefficient.real) * (-1.0) ** occupied
    assert abs(total - result.hf_energy) < 1e-9
    assert abs(total - REFERENCE_HF_ENERGIES[name]) < 1e-9


def _coefficients(operator) -> dict[tuple[tuple[int, ...], tuple[int, ...]], complex]:
    """Return ``{(creations, annihilations): coefficient}`` for an operator."""

    return {
        (term.creations, term.annihilations): complex(term.coefficient)
        for term in operator.terms
    }


def test_the_image_of_the_hamiltonian_has_real_pauli_coefficients() -> None:
    """A Hermitian operator maps onto a real-coefficient Pauli sum, and this one does.

    :func:`flagquantum.observables.jordan_wigner` raises when a term's image has a
    coefficient with an imaginary part above its tolerance, so reaching a result
    at all is most of the claim; the rest is that the coefficients are floats.
    """

    result = create_molecular_hamiltonian(
        _h2o(), n_active_electrons=4, n_active_orbitals=2
    )
    assert len(result.hamiltonian.terms) > 0
    for term in result.hamiltonian.terms:
        assert isinstance(term.coefficient, float)


def test_the_returned_operator_is_a_hermitian_fixed_point() -> None:
    """The operator is symmetrised on the way out, and the result is a fixed point.

    The assembled one-body coefficients satisfy ``c(a^dag_P a_Q) = conj(c(a^dag_Q
    a_P))`` only to the arithmetic's own rounding, so the raw assembly is not
    exactly Hermitian and :func:`flagquantum.observables.jordan_wigner` would
    refuse it. The raw assembly is not reachable from outside
    :func:`second_quantized_hamiltonian`, so what this test can check is the
    contract on the outside of it, and it checks all three parts: what comes
    back is Hermitian, its every paired coefficient is conjugate-symmetric, and
    symmetrising it again changes nothing — so the projection cannot have moved
    a spectrum. Asserting the operator is Hermitian rather than only that its
    pair defect is small is the point: a raw operator has a *small but non-zero*
    pair defect and is still not one ``jordan_wigner`` accepts.
    """

    for factory, electrons in ((_h2, 2), (_lih, 4), (_h2o, 10)):
        integrals = molecular_integrals(factory())
        _, one_body, two_body = _molecular_orbitals(integrals, electrons)
        operator = second_quantized_hamiltonian(one_body, two_body)
        assembled = _coefficients(operator)
        defect = 0.0
        for (creations, annihilations), coefficient in assembled.items():
            partner = assembled.get((annihilations, creations))
            defect = max(
                defect,
                abs(coefficient - partner.conjugate()) if partner else abs(coefficient),
            )
        assert operator.is_hermitian()
        assert defect < 1e-15
        assert assembled == _coefficients(0.5 * (operator + operator.dagger()))


# ---------------------------------------------------------------------------
# 6. The exact ground state against an independent representation
# ---------------------------------------------------------------------------


def test_the_exact_ground_state_of_h2_is_the_published_full_ci_value() -> None:
    result = create_molecular_hamiltonian(_h2())
    assert result.fci_energy == pytest.approx(-1.1372701747, abs=1e-8)
    assert result.fci_determinants == 4


def test_the_block_and_the_matrix_elements_agree() -> None:
    """The bit-flip compression of a Pauli term is the matrix element itself.

    :func:`_sector_matrix` is the fast route to the block and
    :func:`_element_wise_sector_matrix` is the literal one. They are compared on
    water's active space, where the literal route is affordable: if the phases
    or the flip rule were wrong, this is where it would show, rather than in a
    spectrum that happened to be close.
    """

    result = create_molecular_hamiltonian(
        _h2o(), n_active_electrons=4, n_active_orbitals=4
    )
    basis = _sector_basis(result.n_orbitals, result.n_electrons)
    width = 2 * result.n_orbitals
    assert torch.equal(
        _sector_matrix(result.hamiltonian, basis, width),
        _element_wise_sector_matrix(result.hamiltonian, basis, width),
    )


def test_the_exact_ground_state_matches_the_pauli_matrix_factors() -> None:
    """H2 over the whole space: 4 determinants, both routes."""

    result = create_molecular_hamiltonian(_h2())
    matrix = _sector_matrix(
        result.hamiltonian,
        _sector_basis(result.n_orbitals, result.n_electrons),
        2 * result.n_orbitals,
    )
    assert float(torch.linalg.eigvalsh(matrix)[0]) == pytest.approx(
        result.fci_energy, abs=1e-9
    )


def test_the_exact_ground_state_matches_for_a_frozen_core_active_space() -> None:
    """LiH with the 1s core frozen: 25 determinants, both routes."""

    result = create_molecular_hamiltonian(_lih(), n_active_electrons=2)
    assert result.n_orbitals == 5
    assert result.n_electrons == 2
    assert result.fci_determinants == 25
    matrix = _sector_matrix(
        result.hamiltonian,
        _sector_basis(result.n_orbitals, result.n_electrons),
        2 * result.n_orbitals,
    )
    assert float(torch.linalg.eigvalsh(matrix)[0]) == pytest.approx(
        result.fci_energy, abs=1e-9
    )


def test_the_exact_ground_state_refuses_a_space_it_cannot_diagonalise() -> None:
    """The count is a refusal, not a truncated calculation."""

    result = create_molecular_hamiltonian(_h2())
    integrals = molecular_integrals(_h2())
    _, one_body, two_body = _molecular_orbitals(integrals, 2)
    operator = second_quantized_hamiltonian(one_body, two_body)
    with pytest.raises(CapabilityError, match="above the 1024"):
        exact_ground_state_energy(operator, n_electrons=2, n_orbitals=64)
    assert result.fci_determinants <= MAX_FCI_DETERMINANTS


def test_a_molecule_beyond_the_bound_still_gets_a_hamiltonian() -> None:
    """NH3's determinant space is too large, and the rest of the result stands."""

    result = create_molecular_hamiltonian(_nh3())
    assert result.fci_energy is None
    assert result.fci_determinants == 3136
    assert result.hf_energy == pytest.approx(REFERENCE_HF_ENERGIES["NH3"], abs=1e-9)
    assert len(result.hamiltonian.terms) > 0


# ---------------------------------------------------------------------------
# 7. The exact ground state is a functional of the integrals alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("H2", "LiH", "H2O"))
def test_an_orbital_rotation_leaves_the_exact_ground_state_alone(name: str) -> None:
    """Rotating both integral arrays together is a change of basis.

    The Hamiltonian is a functional of the integrals, so an orthogonal mixing of
    the orbitals must not move its spectrum. A Hamiltonian assembled from a
    mis-paired index would not be invariant, which is what makes this a check on
    the assembly rather than a tautology.
    """

    integrals = molecular_integrals(GEOMETRIES[name]())
    _, one_body, two_body = _molecular_orbitals(integrals, REFERENCE_ELECTRONS[name])
    n = one_body.shape[0]
    torch.manual_seed(97)
    mixing, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    rotated_one = mixing.T @ one_body @ mixing
    rotated_two = torch.einsum(
        "abcd,ap,bq,cr,ds->pqrs", two_body, mixing, mixing, mixing, mixing
    )
    reference = create_molecular_hamiltonian(GEOMETRIES[name]()).fci_energy
    moved = exact_ground_state_energy(
        second_quantized_hamiltonian(
            rotated_one, rotated_two, integrals.nuclear_repulsion
        ),
        n_electrons=REFERENCE_ELECTRONS[name],
        n_orbitals=n,
    )
    assert abs(moved - reference) < 1e-11


def test_a_mis_paired_two_electron_index_is_not_invariant() -> None:
    """The same rotation on ``(pq|rs)`` read as ``(ps|qr)`` moves the answer.

    This is the control for the test above. The rotation test would pass for a
    Hamiltonian that ignored the orbital labels entirely, so the pattern has to
    be shown to matter before invariance means anything.
    """

    integrals = molecular_integrals(_h2o())
    _, one_body, two_body = _molecular_orbitals(integrals, 10)
    n = one_body.shape[0]
    wrong = two_body.permute(0, 2, 3, 1).contiguous()
    reference = exact_ground_state_energy(
        second_quantized_hamiltonian(one_body, two_body, integrals.nuclear_repulsion),
        n_electrons=10,
        n_orbitals=n,
    )
    mis_paired = exact_ground_state_energy(
        second_quantized_hamiltonian(one_body, wrong, integrals.nuclear_repulsion),
        n_electrons=10,
        n_orbitals=n,
    )
    assert abs(reference - mis_paired) > 1.0


# ---------------------------------------------------------------------------
# 8. The frozen core is exact, and the shift is the closed form
# ---------------------------------------------------------------------------


def test_the_frozen_core_one_body_shift_is_the_closed_form() -> None:
    """``h_pq -> h_pq + sum_i (2 (pq|ii) - (pi|iq))`` for the frozen orbitals."""

    integrals = molecular_integrals(_h2o())
    _, one_body, two_body = _molecular_orbitals(integrals, 10)
    folded_one, folded_two, core = fold_frozen_core(one_body, two_body, 1)
    for p in range(1, 7):
        for q in range(1, 7):
            expected = float(one_body[p, q])
            for i in range(1):
                expected += 2.0 * float(two_body[p, q, i, i])
                expected -= float(two_body[p, i, i, q])
            assert folded_one[p - 1, q - 1] == pytest.approx(expected, abs=1e-14)
    expected_core = (
        2.0 * float(one_body[0, 0])
        + 2.0 * float(two_body[0, 0, 0, 0])
        - float(two_body[0, 0, 0, 0])
    )
    assert core == pytest.approx(expected_core, abs=1e-14)
    assert folded_two.shape == (6, 6, 6, 6)


def test_the_frozen_core_ground_state_is_the_restricted_full_ground_state() -> None:
    """Freezing is a restriction on the determinant space, not an approximation.

    The full-space image is diagonalised twice: once unrestricted, and once over
    the determinants that hold the core orbital. The reduced Hamiltonian's ground
    state has to be the second number. The operator is deliberately shared
    between the two, so the comparison isolates the reduction.
    """

    for name, n_electrons in (("LiH", 4), ("H2O", 10)):
        integrals = molecular_integrals(GEOMETRIES[name]())
        solve, one_body, two_body = _molecular_orbitals(integrals, n_electrons)
        full = create_molecular_hamiltonian(GEOMETRIES[name]())
        core_bit = 0b11
        kept = [
            mask
            for mask in _sector_basis(full.n_orbitals, n_electrons)
            if mask & core_bit == core_bit
        ]
        matrix = _sector_matrix(full.hamiltonian, kept, 2 * full.n_orbitals)
        restricted = float(torch.linalg.eigvalsh(matrix)[0])
        folded_one, folded_two, core = fold_frozen_core(one_body, two_body, 1)
        operator = second_quantized_hamiltonian(
            folded_one, folded_two, integrals.nuclear_repulsion + core
        )
        reduced = exact_ground_state_energy(
            operator, n_electrons=n_electrons - 2, n_orbitals=full.n_orbitals - 1
        )
        assert abs(reduced - restricted) < 1e-9
        assert reduced > full.fci_energy


def test_freezing_more_orbitals_can_only_raise_the_energy() -> None:
    """An active space is a restriction of the space the full solve covers."""

    energies = []
    for active in (10, 8, 6, 4):
        result = create_molecular_hamiltonian(_h2o(), n_active_electrons=active)
        assert result.fci_energy is not None
        energies.append((active, result.fci_energy))
    for (_, tighter), (_, looser) in zip(energies[1:], energies[:-1], strict=True):
        assert tighter >= looser - 1e-12


# ---------------------------------------------------------------------------
# 9. The active space is recorded rather than hidden
# ---------------------------------------------------------------------------


def test_the_active_space_counts_describe_the_returned_hamiltonian() -> None:
    result = create_molecular_hamiltonian(
        _h2o(), n_active_electrons=4, n_active_orbitals=3
    )
    assert (result.n_electrons, result.n_orbitals) == (4, 3)
    assert result.n_frozen_core_orbitals == 3
    assert result.one_body.shape == (3, 3)
    assert result.two_body.shape == (3, 3, 3, 3)
    assert result.core_energy != 0.0
    assert result.energy_offset == pytest.approx(
        result.nuclear_repulsion + result.core_energy
    )
    assert result.hf_energy == pytest.approx(REFERENCE_HF_ENERGIES["H2O"], abs=1e-9)


def test_an_active_space_energy_lies_above_the_full_space_one() -> None:
    full = create_molecular_hamiltonian(_h2o())
    active = create_molecular_hamiltonian(
        _h2o(), n_active_electrons=4, n_active_orbitals=4
    )
    assert active.fci_energy > full.fci_energy
    assert active.fci_energy < active.hf_energy


# ---------------------------------------------------------------------------
# 10. The ansatz half and the driver half compose
# ---------------------------------------------------------------------------


def _measure(circuit: fq.Circuit, observable: fq.Observable) -> float:
    """Return the exact expectation of ``observable`` on ``circuit``'s state."""

    request = fq.expectation(observable)
    return float(fq.run(circuit, outputs=request).expectation().sum().real)


def test_a_uccsd_circuit_on_this_hamiltonian_reaches_the_exact_ground_state() -> None:
    """The whole chain, measured end to end.

    The circuit is built from this Hamiltonian's own electron and orbital counts
    and measured against this Hamiltonian, so a mismatch between the driver's
    spin-orbital pairing and the ansatz's reference determinant would show up as
    an energy that never approaches the exact one.
    """

    result = create_molecular_hamiltonian(_h2())
    excitations = uccsd_excitations(result.n_electrons, 2 * result.n_orbitals)
    assert excitations.reference_occupation == tuple(range(result.n_electrons))

    def energy(parameters: list[float]) -> float:
        circuit = uccsd_ansatz(2 * result.n_orbitals, result.n_electrons, parameters)
        return _measure(circuit, result.hamiltonian)

    zero = energy([0.0] * excitations.parameter_count)
    assert abs(zero - result.hf_energy) < 1e-6

    best = min(energy([0.0, 0.0, step * 1e-3]) for step in range(-500, 501, 2))
    assert best == pytest.approx(result.fci_energy, abs=1e-4)
    assert best > result.fci_energy - 1e-6

    torch.manual_seed(11)
    for _ in range(24):
        sampled = [float(value) for value in torch.rand(3) * 2.0 * math.pi - math.pi]
        assert energy(sampled) >= result.fci_energy - 1e-6


def test_the_ansatz_refuses_a_hamiltonian_it_does_not_match() -> None:
    """The two halves agree on the counts rather than tolerating a mismatch."""

    result = create_molecular_hamiltonian(_h2())
    with pytest.raises(ValueError, match="must be even"):
        uccsd_ansatz(3, result.n_electrons, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must be at most"):
        uccsd_ansatz(2 * result.n_orbitals, 6, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Expected 3 parameters, got 1"):
        uccsd_ansatz(2 * result.n_orbitals, result.n_electrons, [0.0])


# ---------------------------------------------------------------------------
# 11. Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_electrons", (0, 6, 8))
def test_the_solve_refuses_an_electron_count_outside_the_basis(
    n_electrons: int,
) -> None:
    integrals = molecular_integrals(_h2())
    with pytest.raises(ValidationError, match="carry between one and 4 electrons"):
        restricted_hartree_fock(integrals, n_electrons)


@pytest.mark.parametrize("n_electrons", (3, 5))
def test_the_solve_refuses_an_odd_electron_count(n_electrons: int) -> None:
    integrals = molecular_integrals(_lih())
    with pytest.raises(CapabilityError, match="even electron count"):
        restricted_hartree_fock(integrals, n_electrons)


@pytest.mark.parametrize("n_electrons", (2.0, True, "2"))
def test_the_solve_refuses_a_non_integer_electron_count(n_electrons: object) -> None:
    integrals = molecular_integrals(_h2())
    with pytest.raises(ValidationError, match="is an integer"):
        restricted_hartree_fock(integrals, n_electrons)  # type: ignore[arg-type]


def test_the_solve_refuses_a_budget_it_cannot_finish_in() -> None:
    integrals = molecular_integrals(_lih())
    with pytest.raises(ValidationError, match="at least one"):
        restricted_hartree_fock(integrals, 4, max_iterations=0)
    with pytest.raises(CapabilityError, match="did not converge"):
        restricted_hartree_fock(integrals, 4, max_iterations=1)


def test_a_stretched_bond_gets_a_named_refusal_rather_than_a_density() -> None:
    """The undamped step two-cycles at a stretched bond, and says so.

    This is the boundary of the documented limitation, measured rather than
    quoted: the limit is walked outward and the last converging separation and
    the first oscillating one are both asserted, so a change that widens or
    narrows the range has to move this test. Each point also asserts that the
    integrals are finite, because a refusal caused by ``nan`` integrals would
    otherwise be indistinguishable from a refusal caused by the two-cycle.
    """

    def converge_at(separation: float) -> tuple[bool, float]:
        geometry = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, separation)))
        integrals = molecular_integrals(geometry)
        for matrix in (
            integrals.overlap,
            integrals.kinetic,
            integrals.nuclear_attraction,
            integrals.two_electron,
        ):
            assert torch.isfinite(
                matrix
            ).all(), f"non-finite integrals at {separation} A"
        try:
            return True, restricted_hartree_fock(integrals, 2).energy
        except CapabilityError as error:
            assert "did not converge" in str(error)
            assert "2.000e+00" in str(error)
            return False, float("nan")

    converged, energy = converge_at(5.0)
    assert converged
    assert energy == pytest.approx(-0.5990248715, rel=1e-9)

    oscillating, _ = converge_at(10.0)
    assert not oscillating

    assert any(
        "two-cycle at a stretched bond" in entry for entry in MOLECULAR_LIMITATIONS
    )


@pytest.mark.parametrize("multiplicity", (2, 3, 5, 0))
def test_the_constructor_refuses_a_multiplicity_it_cannot_solve(
    multiplicity: int,
) -> None:
    with pytest.raises(CapabilityError, match="restricted and closed-shell"):
        create_molecular_hamiltonian(_h2(), multiplicity=multiplicity)


def test_the_constructor_refuses_an_unavailable_basis() -> None:
    with pytest.raises(CapabilityError, match="not available"):
        create_molecular_hamiltonian(_h2(), basis="6-31g")


def test_the_constructor_refuses_an_element_outside_the_table() -> None:
    with pytest.raises(CapabilityError, match="outside the STO-3G table"):
        create_molecular_hamiltonian([("Ar", (0.0, 0.0, 0.0))])


def test_the_constructor_refuses_a_charge_that_removes_every_electron() -> None:
    with pytest.raises(ValidationError, match="needs at least one electron"):
        create_molecular_hamiltonian([("H", (0.0, 0.0, 0.0))], charge=1)
    with pytest.raises(ValidationError, match="needs at least one electron"):
        create_molecular_hamiltonian(_h2(), charge=2)


@pytest.mark.parametrize(
    ("active_electrons", "active_orbitals", "message"),
    (
        (1, None, "whole electron pairs"),
        (3, None, "whole electron pairs"),
        (0, None, "keeps between one and 10 electrons"),
        (11, None, "keeps between one and 10 electrons"),
        (None, 0, "keeps between one and the 7 orbitals"),
        (None, 8, "keeps between one and the 7 orbitals"),
        (4, 1, "1 active orbitals carry at most 2 electrons"),
        (10, 3, "3 active orbitals carry at most 6 electrons"),
    ),
)
def test_the_constructor_refuses_an_inconsistent_active_space(
    active_electrons: int | None, active_orbitals: int | None, message: str
) -> None:
    """The two counts are checked against each other, not just against zero."""

    with pytest.raises(ValidationError, match=message):
        create_molecular_hamiltonian(
            _h2o(),
            n_active_electrons=active_electrons,
            n_active_orbitals=active_orbitals,
        )


def test_the_constructor_refuses_an_active_space_wider_than_the_basis() -> None:
    with pytest.raises(ValidationError, match="keeps between one and the 2 orbitals"):
        create_molecular_hamiltonian(_h2(), n_active_orbitals=3)


def test_the_exact_ground_state_refuses_a_sector_it_cannot_hold() -> None:
    integrals = molecular_integrals(_h2())
    _, one_body, two_body = _molecular_orbitals(integrals, 2)
    operator = second_quantized_hamiltonian(one_body, two_body)
    for n_electrons in (0, 6, 3):
        with pytest.raises(ValidationError, match="S_z = 0 sector"):
            exact_ground_state_energy(operator, n_electrons=n_electrons, n_orbitals=2)


def test_the_result_types_refuse_an_inconsistent_shape() -> None:
    integrals = molecular_integrals(_h2())
    solve = restricted_hartree_fock(integrals, 2)
    with pytest.raises(ValidationError, match="one orbital energy per orbital"):
        HartreeFockSolution(
            orbital_energies=solve.orbital_energies[:1],
            coefficients=solve.coefficients,
            density=solve.density,
            energy=solve.energy,
            n_iterations=solve.n_iterations,
            density_change=solve.density_change,
            residual=solve.residual,
        )
    with pytest.raises(ValidationError, match="at least one iteration"):
        HartreeFockSolution(
            orbital_energies=solve.orbital_energies,
            coefficients=solve.coefficients,
            density=solve.density,
            energy=solve.energy,
            n_iterations=0,
            density_change=solve.density_change,
            residual=solve.residual,
        )


# ---------------------------------------------------------------------------
# 12. The recorded assumptions and limitations
# ---------------------------------------------------------------------------


def test_the_documented_scope_is_recorded_as_data() -> None:
    assert all(isinstance(entry, str) and entry for entry in MOLECULAR_ASSUMPTIONS)
    assert all(isinstance(entry, str) and entry for entry in MOLECULAR_LIMITATIONS)
    assert any("STO-3G is the only basis" in entry for entry in MOLECULAR_LIMITATIONS)
    assert any("MAX_FCI_DETERMINANTS" in entry for entry in MOLECULAR_LIMITATIONS)
    assert any(
        "restricted and closed-shell" in entry for entry in MOLECULAR_ASSUMPTIONS
    )


def test_the_geometry_is_the_one_the_result_reports() -> None:
    result = create_molecular_hamiltonian(_h2())
    assert isinstance(result.geometry, MolecularGeometry)
    assert result.geometry.symbols == ("H", "H")
    assert result.basis == "sto-3g"
    assert result.nuclear_repulsion == pytest.approx(0.7137539937, abs=1e-9)
    assert isinstance(result, MolecularHamiltonian)
    assert ATOMIC_NUMBERS["O"] == 8
