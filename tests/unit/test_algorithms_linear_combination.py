"""What the linear-combination block encoding guarantees, and where it stops.

The spectral encoding's test file proves the protocol by reading a dense matrix's block out
of a circuit. This file proves the second implementation behind the same protocol, and the
properties it asserts are the ones that separate a linear combination of unitaries from a
matrix decomposition: the subnormalisation is a sum of coefficient magnitudes the caller's
data already determines, the sign of a coefficient is a diagonal operator rather than a
phase on a prepared amplitude, and the selection is one multi-controlled Pauli product per
term instead of one dense gate.

Three of the assertions here exist because the obvious construction of each is wrong, and a
test that only checked the happy path would pass both:

* The preparation primitive carries no relative phase, so signing the amplitudes cannot
  produce ``H``; the block assertion fails by ``1.333`` on the three-term case if the sign is
  folded into the amplitudes, which is why the sign is a separate diagonal and why the
  signed cases here are the point rather than an extra.
* ``Q`` and ``Sigma`` are each self-adjoint but do not commute as operators, so the adjoint
  walk step has to reverse both products' terms and not merely swap the two products.
  Measured at ``k = 3``, swapping alone leaves a residual of ``1.76``.
* The walk identity holds on the subspace whose ladder wires are ``|0>`` and not on the
  whole register, because the ``2 ** k - m`` index states the preparation leaves empty are
  still states of the register. The walk assertion is written against that subspace, and the
  companion assertion records that the full spectrum has more eigenvalues than the identity
  accounts for.

Tolerances: a gate is stored in the circuit as a complex64 matrix, so every comparison taken
through a circuit is bounded by ``1e-6``; the construction's own dense arithmetic is double
precision and uses ``1e-12``.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives import (
    BlockEncoding,
    LinearCombinationEncoding,
    WalkEncoding,
)
from flagquantum.circuit import Circuit
from flagquantum.errors import CapabilityError
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

# A gate is stored in the circuit as a complex64 matrix, so a comparison taken through a
# circuit cannot be tighter than that.
_CIRCUIT_TOLERANCE = 1e-6

# The construction's own dense arithmetic is double precision.
_EXACT_TOLERANCE = 1e-12

_PAULI = {
    "x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}

# One case per structural property: a two-term one-qubit sum whose index register is a single
# wire, a signed three-term sum which is the smallest case with a sign diagonal and no ladder,
# a four-term two-qubit sum, and the six- and eight-term two-qubit sums that first need a
# ladder ancilla at ``k = 3``. The last two are the boundary cases: an even term count and one
# that fills the index register exactly.
_CASES = (
    ("two_terms_one_qubit", ((1.0, ((0, "x"),)), (0.5, ((0, "z"),))), 1),
    (
        "three_terms_signed",
        ((1.0, ((0, "x"),)), (0.5, ((0, "z"),)), (-1.5, ((0, "y"),))),
        1,
    ),
    (
        "four_terms_two_qubits",
        (
            (1.0, ((0, "x"),)),
            (0.5, ((1, "z"),)),
            (-1.5, ((0, "y"), (1, "x"))),
            (0.25, ((0, "z"), (1, "z"))),
        ),
        2,
    ),
    (
        "six_terms_two_qubits",
        (
            (1.0, ((0, "x"),)),
            (0.5, ((1, "z"),)),
            (-1.5, ((0, "y"), (1, "x"))),
            (0.25, ((0, "z"), (1, "z"))),
            (0.75, ((0, "x"), (1, "x"))),
            (-0.5, ((0, "y"), (1, "y"))),
        ),
        2,
    ),
    (
        "eight_terms_two_qubits",
        (
            (1.0, ((0, "x"),)),
            (0.5, ((1, "z"),)),
            (-1.5, ((0, "y"), (1, "x"))),
            (0.25, ((0, "z"), (1, "z"))),
            (0.75, ((0, "x"), (1, "x"))),
            (-0.5, ((0, "y"), (1, "y"))),
            (0.3, ((0, "z"),)),
            (-0.2, ((1, "x"),)),
        ),
        2,
    ),
)

# The shape of a case: a label, the weighted Pauli words, and the operator's width. A
# coefficient is deliberately ``Any`` because several tests hand the value object something
# that is not a number precisely to check that it is refused.
_Term = tuple[Any, tuple[tuple[int, str], ...]]
_Case = tuple[str, tuple[_Term, ...], int]

_CASE_IDS = [case[0] for case in _CASES]


def _coefficient(term: HamiltonianTerm) -> float:
    """Read a term's coefficient as the Python float the construction reads it as."""
    coefficient: Any = term.coefficient
    return float(coefficient)


def _hamiltonian(terms: tuple[_Term, ...]) -> Hamiltonian:
    """Build the ``Hamiltonian`` a case names.

    The construction reads ``term.ops``, which is the normalised non-identity reading the
    value object caches, so the mapping form is used and a term with no non-identity factor
    is written as an empty mapping rather than as an ``"i"`` a reader would have to strip.
    """
    return Hamiltonian(
        [
            HamiltonianTerm(coefficient, dict(ops), sorted(dict(ops)) or None)
            for coefficient, ops in terms
        ]
    )


def _dense(hamiltonian: Hamiltonian, n_system: int) -> torch.Tensor:
    """Return the Pauli sum as a dense matrix, rebuilt from the Pauli words."""
    identity = torch.eye(2, dtype=torch.complex128)
    total = torch.zeros(1 << n_system, 1 << n_system, dtype=torch.complex128)
    for term in hamiltonian.terms:
        block = torch.tensor([[1.0 + 0.0j]], dtype=torch.complex128)
        for wire in range(n_system):
            axis = next((name for w, name in term.ops if w == wire), None)
            block = torch.kron(block, identity if axis is None else _PAULI[axis])
        total = total + _coefficient(term) * block
    return total


def _flagged_block(encoding: LinearCombinationEncoding, n_system: int) -> torch.Tensor:
    """Read the encoding's flagged block out of the circuit, column by column.

    The block is ``<0|_flag U |0>_flag``, so its column ``j`` is what the circuit leaves on
    the operator's wires when it is entered with the basis state ``|j>`` and the whole flag
    register left in ``|0>``. The circuit is entered with only the operator's wires set, so
    every flagged amplitude is a leading one and the block is the leading square of each
    column. Nothing here reads the unitary: what is read is the state the composition leaves.
    """
    dimension = 1 << n_system
    system = list(range(encoding.num_ancilla, encoding.num_ancilla + n_system))
    columns = []
    for basis in range(dimension):
        circuit = Circuit(encoding.num_ancilla + n_system)
        for index in range(n_system):
            if (basis >> (n_system - 1 - index)) & 1:
                circuit.gate("x", system[index])
        encoding.append_apply(circuit, ancilla=0, qubits=system)
        columns.append(circuit.state().reshape(-1)[:dimension])
    return torch.stack(columns, dim=1)


def _ladder_zero_subspace(
    encoding: LinearCombinationEncoding, n_system: int
) -> torch.Tensor:
    """Return the state indices whose ladder wires are all ``|0>``.

    The ladder ancillas are the last ``num_ancilla - index_width`` wires of the flag register,
    and a higher wire index is a less significant bit, so each one names a bit of the address
    found from the circuit's width rather than assumed to be a low one.
    """
    n_wires = encoding.num_ancilla + n_system
    mask = 0
    for wire in range(encoding.index_width, encoding.num_ancilla):
        mask |= 1 << (n_wires - 1 - wire)
    return torch.tensor(
        [state for state in range(1 << n_wires) if state & mask == 0], dtype=torch.long
    )


def _cosines(
    encoding: LinearCombinationEncoding, n_system: int, *, walk: bool
) -> torch.Tensor:
    """Return the cosines of a circuit's eigenphases, sorted ascending.

    This is a consumer of the boundary and not of an implementation: it names only
    ``num_ancilla``, ``index_width``, ``append_apply`` and ``append_walk_step``. With
    ``walk`` it appends the phase flip as well, so the two readings differ by exactly the
    reflection and nothing else.
    """
    n_ancilla = encoding.num_ancilla
    circuit = Circuit(n_ancilla + n_system)
    qubits = list(range(n_ancilla, n_ancilla + n_system))
    if walk:
        encoding.append_walk_step(circuit, ancilla=0, qubits=qubits)
    else:
        encoding.append_apply(circuit, ancilla=0, qubits=qubits)
    unitary = get_unitary(circuit, dtype=torch.complex128)
    subset = _ladder_zero_subspace(encoding, n_system)
    unitary = unitary[subset][:, subset]
    return torch.sort(torch.cos(torch.angle(torch.linalg.eigvals(unitary)))).values


def _requested_spectrum(
    encoding: LinearCombinationEncoding, n_system: int
) -> torch.Tensor:
    """Return ``H / alpha``'s eigenvalues, sorted ascending.

    ``eigvalsh`` is a different routine from anything the construction uses -- the
    construction never diagonalises the operator at all -- and it is applied to a dense
    matrix rebuilt from the Pauli words rather than to one the encoding holds, so the expected
    side of the walk identity is not the construction's own intermediate value.
    """
    matrix = _dense(encoding.hamiltonian, n_system) / encoding.alpha
    return torch.sort(torch.linalg.eigvalsh(matrix).to(torch.float64)).values


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_the_flagged_block_is_the_pauli_sum_over_the_sum_of_magnitudes(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """The flagged block of the emitted circuit is ``H / alpha``, read out of the circuit."""
    hamiltonian = _hamiltonian(terms)
    encoding = LinearCombinationEncoding(hamiltonian)
    block = _flagged_block(encoding, n_system)
    wanted = _dense(hamiltonian, n_system) / encoding.alpha
    assert float((block - wanted).abs().max()) < _CIRCUIT_TOLERANCE


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_the_subnormalisation_is_the_sum_of_the_coefficient_magnitudes(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """``alpha`` is a number the caller's data determines, not a norm read off a matrix.

    It is at least the spectral norm, so every eigenvalue of the encoded block lies in
    ``[-1, 1]`` -- the condition that makes this a block encoding rather than a scaled
    operator. It is *not* the Frobenius norm, and the two are not even ordered: the Frobenius
    norm of a Pauli product on ``n`` wires is ``2 ** (n / 2)``, so a one-wire sum can have a
    Frobenius norm above the sum of its coefficient magnitudes. That is why
    :func:`~flagquantum.algorithms.primitives.block_encoding.subnormalisation` is not reused
    here.
    """
    hamiltonian = _hamiltonian(terms)
    encoding = LinearCombinationEncoding(hamiltonian)
    magnitudes = [abs(_coefficient(term)) for term in hamiltonian.terms]
    assert encoding.alpha == pytest.approx(math.fsum(magnitudes), abs=_EXACT_TOLERANCE)
    matrix = _dense(hamiltonian, n_system)
    spectral = float(torch.linalg.eigvalsh(matrix).abs().max())
    assert encoding.alpha >= spectral - _EXACT_TOLERANCE


def test_the_block_is_not_the_matrix_itself_and_the_factor_is_not_the_frobenius_norm() -> (
    None
):
    """The encoding normalises, and it does not normalise by the norm a matrix would give.

    A construction that returned ``H`` unchanged would satisfy every "the block is close to
    something" assertion above if the expected side were also ``H``. The factor is separated
    from the Frobenius norm on the smallest sum where the two differ in either direction:
    one term on one wire is ``1.5 X``, whose Frobenius norm is ``1.5 sqrt(2)`` -- larger than
    the factor, because the Frobenius norm of a Pauli product on ``n`` wires is
    ``2 ** (n / 2)``. Reusing
    :func:`~flagquantum.algorithms.primitives.block_encoding.subnormalisation` would pick
    that larger number for a matrix it would have to form first.
    """
    hamiltonian = _hamiltonian(((1.5, ((0, "x"),)),))
    encoding = LinearCombinationEncoding(hamiltonian)
    matrix = _dense(hamiltonian, 1)
    block = _flagged_block(encoding, 1)
    frobenius = float(torch.linalg.matrix_norm(matrix))
    assert encoding.alpha == pytest.approx(1.5, abs=_EXACT_TOLERANCE)
    assert frobenius == pytest.approx(1.5 * math.sqrt(2.0), abs=_EXACT_TOLERANCE)
    assert frobenius > encoding.alpha + 1e-3
    assert float((block - matrix).abs().max()) > 1e-3


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_a_signed_sum_encodes_the_sum_and_not_its_magnitudes(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """The sign of a coefficient is carried, and it is carried by a diagonal operator.

    The preparation primitive this module uses realises the amplitudes it is handed up to one
    phase for the whole vector, so it cannot carry a relative sign: signed amplitudes would
    encode ``sum |c_j| P_j`` and not ``sum c_j P_j``. This assertion is what fails, by
    ``1.333`` on the three-term case, if the sign is folded into the amplitudes instead.
    """
    hamiltonian = _hamiltonian(terms)
    encoding = LinearCombinationEncoding(hamiltonian)
    block = _flagged_block(encoding, n_system)
    wanted = _dense(hamiltonian, n_system) / encoding.alpha
    unsigned = (
        _dense(_hamiltonian(tuple((abs(c), ops) for c, ops in terms)), n_system)
        / encoding.alpha
    )
    assert float((block - wanted).abs().max()) < _CIRCUIT_TOLERANCE
    if any(coefficient < 0.0 for coefficient, _ in terms):
        assert float((block - unsigned).abs().max()) > 1e-3


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_the_flag_register_is_the_index_wires_and_the_ladder(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """``num_ancilla`` is ``k + max(k - 2, 0)`` and ``num_system`` comes from the words."""
    encoding = LinearCombinationEncoding(_hamiltonian(terms))
    expected_k = max(1, (len(terms) - 1).bit_length())
    assert encoding.index_width == expected_k
    assert encoding.num_ancilla == expected_k + max(expected_k - 2, 0)
    assert encoding.num_system == n_system


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_the_encoding_satisfies_both_protocols_by_surface(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """The second implementation is admitted by the protocols without inheriting them."""
    encoding = LinearCombinationEncoding(_hamiltonian(terms))
    assert isinstance(encoding, BlockEncoding)
    assert isinstance(encoding, WalkEncoding)


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_the_encoding_is_a_unitary_and_its_adjoint_walk_step_inverts_it(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """The encoding is unitary, and the adjoint walk step is its inverse to the last bit.

    The adjoint is asserted entrywise against the identity rather than on the flagged
    subspace, because that is the assertion that separates a reversed select from a reused
    one: on the six-term sum the select's terms in reverse differ from the forward order by
    ``1.66``, so a residual of that size is what a reused select leaves. Which case first
    needs the reversal, and why, is asserted separately.
    """
    encoding = LinearCombinationEncoding(_hamiltonian(terms))
    n_ancilla = encoding.num_ancilla
    n_wires = n_ancilla + n_system
    qubits = list(range(n_ancilla, n_wires))
    identity = torch.eye(1 << n_wires, dtype=torch.complex128)

    encoded = Circuit(n_wires)
    encoding.append_apply(encoded, ancilla=0, qubits=qubits)
    unitary = get_unitary(encoded, dtype=torch.complex128)
    assert float((unitary @ unitary.mH - identity).abs().max()) < _CIRCUIT_TOLERANCE

    step = Circuit(n_wires)
    encoding.append_walk_step(step, ancilla=0, qubits=qubits)
    backward = Circuit(n_wires)
    encoding.append_adjoint_walk_step(backward, ancilla=0, qubits=qubits)
    forward = get_unitary(step, dtype=torch.complex128)
    reverse = get_unitary(backward, dtype=torch.complex128)
    assert float((forward @ reverse - identity).abs().max()) < _CIRCUIT_TOLERANCE
    assert float((reverse @ forward - identity).abs().max()) < _CIRCUIT_TOLERANCE


def test_the_adjoint_reverses_the_select_because_the_encoding_is_not_its_own_adjoint() -> (
    None
):
    """Why the adjoint cannot reuse the forward select, told on the first case that needs one.

    The select's terms share the ladder ancillas the multi-controlled gates are built from,
    so their product is not its own reverse even though each term's factor is Hermitian. The
    encoding therefore fails to be Hermitian by ``1.664`` on the six-term sum -- the first
    case whose index register is wide enough to need a ladder -- and the walk step is not its
    own inverse either. Reusing the forward select in the adjoint leaves a residual of that
    size, which is what the module docstring records. On the narrower index registers the
    reversal is a no-op and the parametrised inverse test above already passes without it,
    which is why this assertion is not folded into that test.
    """
    encoding = LinearCombinationEncoding(_hamiltonian(_CASES[3][1]))
    assert encoding.index_width == 3
    n_wires = encoding.num_ancilla + 2
    qubits = list(range(encoding.num_ancilla, n_wires))
    identity = torch.eye(1 << n_wires, dtype=torch.complex128)

    circuit = Circuit(n_wires)
    encoding.append_apply(circuit, ancilla=0, qubits=qubits)
    unitary = get_unitary(circuit, dtype=torch.complex128)
    assert float((unitary - unitary.mH).abs().max()) > _CIRCUIT_TOLERANCE

    step = Circuit(n_wires)
    encoding.append_walk_step(step, ancilla=0, qubits=qubits)
    forward = get_unitary(step, dtype=torch.complex128)
    assert float((forward @ forward - identity).abs().max()) > _CIRCUIT_TOLERANCE


@pytest.mark.parametrize(("label", "terms", "n_system"), _CASES, ids=_CASE_IDS)
def test_one_walk_step_carries_every_encoded_eigenvalue_twice(
    label: str, terms: tuple[_Term, ...], n_system: int
) -> None:
    """Every ``lambda_j / alpha`` is attained, with multiplicity two, and nothing is missed.

    The subspace is the one whose ladder wires are ``|0>``, which is where the encoding is
    defined. Each distinct eigenvalue of ``H / alpha`` must appear as the cosine of a walk
    eigenvalue once per eigenvector it has, that is twice its degeneracy, and the count must
    match exactly rather than be a lower bound.
    """
    encoding = LinearCombinationEncoding(_hamiltonian(terms))
    cosines = _cosines(encoding, n_system, walk=True)
    wanted = _requested_spectrum(encoding, n_system)
    for value in torch.unique(wanted).tolist():
        degeneracy = int(((wanted - value).abs() < 1e-9).sum())
        assert int(((cosines - value).abs() < 1e-6).sum()) == 2 * degeneracy, (
            label,
            value,
        )


def test_the_reflection_is_a_construction_and_the_bare_encoding_has_another_spectrum() -> (
    None
):
    """Without the phase flip the spectrum is a different set of cosines.

    The sibling the identity above needs. The two readings differ by exactly the reflection
    and nothing else, so if the flip cancelled or were a global phase the two would agree;
    they do not.
    """
    encoding = LinearCombinationEncoding(_hamiltonian(_CASES[3][1]))
    assert (
        float(
            (_cosines(encoding, 2, walk=True) - _cosines(encoding, 2, walk=False))
            .abs()
            .max()
        )
        > 1e-3
    )


def test_the_full_register_spectrum_has_more_eigenvalues_than_the_identity_accounts_for() -> (
    None
):
    """The walk identity is a subspace statement, and the extra eigenvalues are recorded.

    The ``2 ** k - m`` index states the preparation leaves empty are still states of the
    register, and on them the walk acts without an encoded eigenvalue behind it. Asserting an
    equality of whole spectra would be false, so the extra count is asserted here rather than
    the difference being left implicit.
    """
    terms = _CASES[3][1]
    encoding = LinearCombinationEncoding(_hamiltonian(terms))
    assert (1 << encoding.index_width) > len(terms)
    n_system = 2
    circuit = Circuit(encoding.num_ancilla + n_system)
    qubits = list(range(encoding.num_ancilla, encoding.num_ancilla + n_system))
    encoding.append_walk_step(circuit, ancilla=0, qubits=qubits)
    full = torch.sort(
        torch.cos(
            torch.angle(
                torch.linalg.eigvals(get_unitary(circuit, dtype=torch.complex128))
            )
        )
    ).values
    wanted = _requested_spectrum(encoding, n_system)
    matched = sum(
        int(((full - value).abs() < 1e-6).sum())
        for value in torch.unique(wanted).tolist()
    )
    # Every value the identity accounts for is still attained twice, and the register carries
    # further eigenvalues beyond them -- which is why the assertion is a count of extras and
    # not an equality of whole spectra.
    assert matched == 2 * wanted.numel()
    assert full.numel() > matched


def test_an_all_identity_term_is_encoded_where_a_trotter_step_refuses_one() -> None:
    """A sum with an identity term is a sum this construction carries.

    ``HamiltonianTerm`` drops the identity from ``ops``, so a term can name no wire at all.
    The term still contributes its magnitude to ``alpha`` and multiplies the identity's own
    block, so the block is the sum over ``alpha`` and the term is not lost.
    """
    hamiltonian = _hamiltonian(((1.0, ()), (0.5, ((0, "x"),))))
    encoding = LinearCombinationEncoding(hamiltonian)
    assert encoding.alpha == pytest.approx(1.5, abs=_EXACT_TOLERANCE)
    assert encoding.num_system == 1
    block = _flagged_block(encoding, 1)
    wanted = _dense(hamiltonian, 1) / encoding.alpha
    assert float((block - wanted).abs().max()) < _CIRCUIT_TOLERANCE


def test_the_encoding_is_frozen_and_the_hamiltonian_is_the_one_it_was_given() -> None:
    """The value object is frozen, and it records which sum it belongs to."""
    hamiltonian = _hamiltonian(_CASES[2][1])
    encoding = LinearCombinationEncoding(hamiltonian)
    assert encoding.hamiltonian is hamiltonian
    assert encoding.terms == tuple(
        (_coefficient(term), term.ops) for term in hamiltonian.terms
    )
    with pytest.raises(FrozenInstanceError):
        encoding.alpha = 1.0  # type: ignore[misc]


def test_a_coefficient_that_requires_a_gradient_is_read_as_a_number() -> None:
    """The factor is a Python float, so no gradient reaches the sum through the encoding.

    A circuit's gates are not differentiable parameters here, so a coefficient tensor must be
    read as a number rather than carried through; a caller who trains a coefficient does so
    through the operator's own expectation path, not through this encoding.
    """
    coefficient = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    encoding = LinearCombinationEncoding(
        Hamiltonian([HamiltonianTerm(coefficient, {0: "x"}, [0])])
    )
    assert isinstance(encoding.alpha, float)
    assert encoding.alpha == pytest.approx(1.0, abs=_EXACT_TOLERANCE)


@pytest.mark.parametrize(
    ("coefficient", "message"),
    (
        ("x", "not a number"),
        (True, "not a number"),
        (float("nan"), "not finite"),
        (float("inf"), "not finite"),
    ),
)
def test_a_coefficient_that_is_not_a_finite_real_number_is_refused(
    coefficient: object, message: str
) -> None:
    """A coefficient the subnormalisation cannot sum is refused where it is read."""
    with pytest.raises(ValueError, match=message):
        LinearCombinationEncoding(_hamiltonian(((coefficient, ((0, "x"),)),)))


def test_a_complex_coefficient_is_refused_by_name_rather_than_reduced() -> None:
    """The construction has no way to carry a relative phase, and says so.

    A silent reduction to the real part would encode a different operator than the caller
    named, which is the failure mode this refusal exists to prevent; the exception is a
    :class:`~flagquantum.errors.CapabilityError` because the gap is a construction this module
    does not have rather than a value that cannot exist.
    """
    with pytest.raises(CapabilityError, match="complex coefficient"):
        LinearCombinationEncoding(_hamiltonian(((1.0 + 0.5j, ((0, "x"),)),)))
    with pytest.raises(CapabilityError, match="complex coefficient"):
        LinearCombinationEncoding(
            _hamiltonian(((torch.tensor(1.0 + 0.5j), ((0, "x"),)),))
        )


def test_a_zero_sum_is_refused_because_there_is_no_block_to_normalise() -> None:
    """Every term cancelling its own weight leaves nothing to normalise by."""
    with pytest.raises(ValueError, match="subnormalisation is zero"):
        LinearCombinationEncoding(
            _hamiltonian(((0.0, ((0, "x"),)), (0.0, ((0, "z"),))))
        )


def test_a_coefficient_holding_more_than_one_value_is_refused() -> None:
    """A term carries one weight, so a coefficient tensor with more than one is refused."""
    with pytest.raises(ValueError, match="holding 2 values"):
        LinearCombinationEncoding(
            _hamiltonian(((torch.tensor([1.0, 2.0]), ((0, "x"),)),))
        )


def test_something_that_is_not_a_hamiltonian_is_refused() -> None:
    """The construction reads ``terms``, and an object without them is refused by name."""
    with pytest.raises(ValueError, match="has no terms"):
        LinearCombinationEncoding(object())  # type: ignore[arg-type]


def test_an_empty_hamiltonian_is_refused_by_the_value_object_before_it_reaches_here() -> (
    None
):
    """``Hamiltonian`` requires a term, so this is the value object's own refusal."""
    with pytest.raises(ValueError, match="at least one term"):
        Hamiltonian([])


def test_a_terms_table_that_is_empty_is_refused_by_name() -> None:
    """An object whose ``terms`` is empty is refused here rather than reaching the factor.

    ``Hamiltonian`` refuses an empty sum at construction, and that refusal is pinned
    above; but the construction reads ``terms`` off whatever it is handed, which is the
    duck-typed path ``test_something_that_is_not_a_hamiltonian_is_refused`` relies on, so
    an object exposing an empty table reaches this module's own check. The factor of an
    empty table sums to zero anyway, so without this the refusal would still happen and
    would say the subnormalisation vanished rather than that there is nothing to encode.
    """

    class _Empty:
        terms: tuple[object, ...] = ()

    with pytest.raises(ValueError, match="carries no terms"):
        LinearCombinationEncoding(_Empty())  # type: ignore[arg-type]


def test_a_term_naming_an_axis_outside_the_pauli_group_is_refused() -> None:
    """The selection applies X, Y and Z, so any other axis is refused where it is read.

    A Hamiltonian normalises its Pauli names, so a term built by hand is the only way to
    reach this: the check exists because the construction indexes a basis-change table by the
    axis and an unlisted one would otherwise emit a controlled identity.
    """
    hamiltonian = _hamiltonian(((1.0, ((0, "x"),)),))
    object.__setattr__(hamiltonian.terms[0], "ops", ((0, "q"),))
    with pytest.raises(ValueError, match="names Pauli axis 'q'"):
        LinearCombinationEncoding(hamiltonian)


@pytest.mark.parametrize(
    ("ancilla", "qubits", "message"),
    (
        (0, [1], "must be 2 wire"),
        (0, [1, 1], "must be distinct"),
        (True, [1, 2], "must be the first wire"),
        (1.0, [2, 3], "must be the first wire"),
        (0, [0, 2], "overlaps the operator's own wires"),
        (0, [2, 7], "reaches wire"),
    ),
)
def test_a_malformed_register_is_refused_and_leaves_the_circuit_empty(
    ancilla: object, qubits: list[int], message: str
) -> None:
    """Every register condition is checked before a gate is emitted.

    The four-term two-qubit case is used throughout: it has ``k = 2`` and so two flag wires,
    which is what makes ``[1]`` too narrow, ``[1, 1]`` a repeat, and ``[0, 2]`` an overlap
    rather than a legitimate register. The circuit is six wires, so ``[2, 7]`` names a wire
    the circuit does not have.
    """
    encoding = LinearCombinationEncoding(_hamiltonian(_CASES[2][1]))
    circuit = Circuit(6)
    with pytest.raises(ValueError, match=message):
        encoding.append_apply(
            circuit, ancilla=ancilla, qubits=qubits  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=message):
        encoding.append_walk_step(
            circuit, ancilla=ancilla, qubits=qubits  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=message):
        encoding.append_adjoint_walk_step(
            circuit, ancilla=ancilla, qubits=qubits  # type: ignore[arg-type]
        )
    assert circuit.to_qir() == []


def test_the_two_registers_need_not_be_adjacent() -> None:
    """The encoding composes with free wires between the two registers.

    The flag register is at the bottom and the operator's single wire is placed two wires
    above it, so nothing here depends on the two registers being adjacent; the encoded block
    is read out of the circuit and is the same block either way.
    """
    hamiltonian = _hamiltonian(_CASES[1][1])
    encoding = LinearCombinationEncoding(hamiltonian)
    n_ancilla = encoding.num_ancilla
    spare = 2
    system = [n_ancilla + spare]
    dimension = 1 << encoding.num_system
    columns = []
    for basis in range(dimension):
        circuit = Circuit(n_ancilla + spare + encoding.num_system)
        for index in range(encoding.num_system):
            if (basis >> (encoding.num_system - 1 - index)) & 1:
                circuit.gate("x", system[index])
        encoding.append_apply(circuit, ancilla=0, qubits=system)
        columns.append(circuit.state().reshape(-1)[:dimension])
    block = torch.stack(columns, dim=1)
    wanted = _dense(hamiltonian, encoding.num_system) / encoding.alpha
    assert float((block - wanted).abs().max()) < _CIRCUIT_TOLERANCE
