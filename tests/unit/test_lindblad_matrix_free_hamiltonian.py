"""The matrix-free Hamiltonian representation for continuous-time evolution.

These tests hold the representation to the dense route it replaces: the Pauli
action must reproduce the dense matrix entry for entry, the commutator must
reproduce ``H @ rho - rho @ H``, and every acceptance decision must match.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import SerializationError
from flagquantum.observables import Observable, _PauliTerm
from flagquantum.operators import SuperOperator
from flagquantum.simulation import (
    EvolutionValidationError,
    amplitude_damping,
    evolve_density_matrix,
    plan_density_matrix_evolution,
)
from flagquantum.simulation import lindblad as lindblad_module
from flagquantum.simulation.density_matrix import expand_operator
from flagquantum.simulation.matrix_free_hamiltonian import PauliSum, PauliSumTerm

pytestmark = pytest.mark.unit

_DEVICE = torch.device("cpu")
_TOLERANCE = 1e-10


def _complex_term(coefficient: complex, factors: Any) -> _PauliTerm:
    """Build a term whose coefficient is complex, which the algebra rejects."""

    return _PauliTerm(cast(Any, coefficient), factors)


def _dense_hamiltonian(hamiltonian: object, *, n_wires: int) -> torch.Tensor:
    """Build the dense matrix the way v0.2.0 did, from untouched primitives.

    The reference deliberately reuses ``_pauli_operator`` and
    ``_descriptor_matrix``, so it fails if the representation and the
    placement convention ever drift apart.
    """

    dim = 2**n_wires
    dtype = torch.complex128
    if isinstance(hamiltonian, Observable):
        matrix = torch.zeros((dim, dim), dtype=dtype, device=_DEVICE)
        identity = torch.eye(dim, dtype=dtype, device=_DEVICE)
        for term in hamiltonian.terms:
            if not term.factors:
                matrix = matrix + term.coefficient * identity
                continue
            matrix = matrix + term.coefficient * lindblad_module._pauli_operator(
                "".join(axis for _, axis in term.factors),
                tuple(wire for wire, _ in term.factors),
                n_wires=n_wires,
                dtype=dtype,
                device=_DEVICE,
                field="hamiltonian",
            )
        return matrix
    if isinstance(hamiltonian, torch.Tensor):
        return lindblad_module._as_complex_matrix(
            hamiltonian,
            name="hamiltonian",
            dim=dim,
            dtype=dtype,
            device=_DEVICE,
        )
    matrix = torch.zeros((dim, dim), dtype=dtype, device=_DEVICE)
    for term in hamiltonian:  # type: ignore[attr-defined]
        matrix = matrix + float(term.get("coefficient", 1.0)) * (
            lindblad_module._descriptor_matrix(
                term,
                n_wires=n_wires,
                dtype=dtype,
                device=_DEVICE,
                field="hamiltonian",
            )
        )
    return matrix


def _dense_trajectory(
    hamiltonian: object,
    n_wires: int,
    times: list[float],
    *,
    collapse_rate: float | None = None,
) -> torch.Tensor:
    """Reproduce the RK4 trajectory from the dense matrix alone."""

    dim = 2**n_wires
    dtype = torch.complex128
    matrix = _dense_hamiltonian(hamiltonian, n_wires=n_wires)
    state = torch.zeros((dim, dim), dtype=dtype, device=_DEVICE)
    state[0, 0] = 1.0
    operators = []
    if collapse_rate is not None:
        sigma_minus = torch.tensor(
            [[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=_DEVICE
        )
        operators.append(
            collapse_rate**0.5
            * expand_operator(sigma_minus, (0,), n_wires, dtype=dtype, device=_DEVICE)
        )
    grid = torch.tensor(times, dtype=torch.float64)
    states = [state]
    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        step = ((stop - start) / 2).to(dtype=torch.float64)
        for _ in range(2):

            def derivative(value: torch.Tensor) -> torch.Tensor:
                total = -1j * (matrix @ value - value @ matrix)
                for operator in operators:
                    total = (
                        total
                        + operator @ value @ operator.conj().T
                        - 0.5
                        * (
                            operator.conj().T @ operator @ value
                            + value @ operator.conj().T @ operator
                        )
                    )
                return total

            k1 = derivative(state)
            k2 = derivative(state + step * k1 / 2)
            k3 = derivative(state + step * k2 / 2)
            k4 = derivative(state + step * k3)
            state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(state)
    return torch.stack(states)


def _normalize(
    hamiltonian: object, *, n_wires: int, dtype: torch.dtype = torch.complex128
) -> PauliSum | torch.Tensor:
    return lindblad_module._normalize_hamiltonian(
        hamiltonian,
        n_wires=n_wires,
        dim=2**n_wires,
        dtype=dtype,
        device=_DEVICE,
        tolerance=_TOLERANCE,
    )


def test_repeated_pauli_strings_are_summed_entry_for_entry() -> None:
    """A string named twice is one term with the summed coefficient.

    Resolving the repeat by keeping either occurrence changes the operator,
    and -- for a cancelling pair -- changes the Hermiticity verdict, because
    each occurrence alone is complex while the sum is real.
    """

    descriptors = [
        {"pauli": "Z", "wires": [0], "coefficient": 0.3},
        {"pauli": "Z", "wires": [0], "coefficient": 0.4},
    ]
    representation = _normalize(descriptors, n_wires=1)
    assert isinstance(representation, PauliSum)
    reference = _dense_hamiltonian(descriptors, n_wires=1)
    assert torch.equal(representation.dense(), reference)
    state = torch.randn(2, 2, dtype=torch.complex128)
    drift = representation.commutator(state) - (reference @ state - state @ reference)
    assert float(drift.abs().max()) < 1e-15

    cancelling = Observable(
        terms=(
            _complex_term(1j, ((0, "z"),)),
            _complex_term(-1j, ((0, "z"),)),
        )
    )
    representation = _normalize(cancelling, n_wires=1)
    assert isinstance(representation, PauliSum)
    assert torch.equal(
        representation.dense(), torch.zeros(2, 2, dtype=torch.complex128)
    )
    assert torch.equal(
        representation.commutator(state), torch.zeros(2, 2, dtype=torch.complex128)
    )


def test_a_complex64_hamiltonian_keeps_the_state_precision() -> None:
    """The cached sign vector must not widen the block working set.

    ``commutator`` writes back into an ``empty_like`` result, so a promoted
    temporary is invisible in the returned values -- but it doubles the bytes
    the block bound is supposed to control, which is the whole point of the
    bound. The block itself is therefore what has to be checked.
    """

    representation = lindblad_module._normalize_hamiltonian(
        0.5 * fq.X(0) + 0.25 * fq.Z(0),
        n_wires=1,
        dim=2,
        dtype=torch.complex64,
        device=_DEVICE,
        tolerance=1e-5,
    )
    assert isinstance(representation, PauliSum)
    state = torch.randn(2, 2, dtype=torch.complex64)
    assert representation.dense().dtype == torch.complex64
    assert representation.commutator(state).dtype == torch.complex64
    assert representation._commutator_block(state, 0, 2).dtype == torch.complex64


@pytest.mark.parametrize("n_wires", [1, 2, 3, 4])
def test_the_pauli_action_reproduces_the_dense_matrix_entry_for_entry(
    n_wires: int,
) -> None:
    """A Y factor must contribute its ``i`` exactly once.

    ``Y = i X Z``, so the bit-flip word a term executes is not the Pauli
    string the caller named. Storing the word's coefficient instead of the
    string's shifts every Y term by a quarter turn and is invisible to a
    Hermiticity check on real coefficients.
    """

    torch.manual_seed(20240617 + n_wires)
    for _ in range(6):
        terms = []
        for _ in range(4):
            weight = int(torch.randint(1, n_wires + 1, (1,)).item())
            wires = sorted(int(wire) for wire in torch.randperm(n_wires)[:weight])
            factors = tuple(
                (wire, "xyz"[int(torch.randint(0, 3, (1,)).item())]) for wire in wires
            )
            terms.append(_PauliTerm(float(torch.randn(1).item()), factors))
        hamiltonian = Observable(terms=tuple(terms))
        representation = _normalize(hamiltonian, n_wires=n_wires)
        assert isinstance(representation, PauliSum)

        reference = _dense_hamiltonian(hamiltonian, n_wires=n_wires)
        assert torch.equal(representation.dense(), reference)

        state = torch.randn(2**n_wires, 2**n_wires, dtype=torch.complex128)
        commutator = representation.commutator(state)
        expected = reference @ state - state @ reference
        assert float((commutator - expected).abs().max()) < 1e-14


def test_a_descriptor_sequence_reaches_the_same_representation() -> None:
    torch.manual_seed(4242)
    for _ in range(5):
        descriptors = []
        for _ in range(3):
            weight = int(torch.randint(1, 4, (1,)).item())
            wires = sorted(int(wire) for wire in torch.randperm(3)[:weight])
            descriptors.append(
                {
                    "pauli": "".join(
                        "xyz"[int(torch.randint(0, 3, (1,)).item())] for _ in wires
                    ),
                    "wires": wires,
                    "coefficient": float(torch.randn(1).item()),
                }
            )
        representation = _normalize(descriptors, n_wires=3)
        assert isinstance(representation, PauliSum)

        reference = _dense_hamiltonian(descriptors, n_wires=3)
        assert float((representation.dense() - reference).abs().max()) < 1e-14

        state = torch.randn(8, 8, dtype=torch.complex128)
        drift = representation.commutator(state) - (
            reference @ state - state @ reference
        )
        assert float(drift.abs().max()) < 1e-14


@pytest.mark.parametrize(
    ("hamiltonian", "n_wires", "collapse_rate"),
    [
        (0.5 * fq.X(0) + 0.1 * fq.Z(0), 1, None),
        (1.5 * fq.Y(0), 1, 0.3),
        (0.4 * (fq.Y(0) @ fq.Y(1)) + 0.9 * fq.X(1), 2, None),
        (
            0.3 * (fq.X(0) @ fq.X(1)) + 0.2 * (fq.Z(0) @ fq.Z(1)) + 0.1 * fq.Z(0),
            2,
            None,
        ),
        (
            [
                {"pauli": "Z", "wires": [0], "coefficient": 0.7},
                {"pauli": "XX", "wires": [0, 1], "coefficient": -0.4},
                {"pauli": "Y", "wires": [1], "coefficient": 1.3},
            ],
            2,
            None,
        ),
        (
            [
                {"pauli": "I", "wires": [0], "coefficient": 2.5},
                {"pauli": "Z", "wires": [1], "coefficient": 0.25},
            ],
            2,
            None,
        ),
        (torch.tensor([[0.5, 0.25j], [-0.25j, -0.5]], dtype=torch.complex128), 1, None),
    ],
)
def test_the_trajectory_matches_the_dense_runge_kutta_reference(
    hamiltonian: object, n_wires: int, collapse_rate: float | None
) -> None:
    """The end-to-end trajectory, not just the derivative, must be unchanged."""

    times = [0.0, 0.05, 0.11, 0.2]
    options = (
        {"collapse_operators": [amplitude_damping(rate=collapse_rate, wire=0)]}
        if collapse_rate is not None
        else {}
    )
    result = evolve_density_matrix(
        hamiltonian,
        "0" * n_wires,
        n_wires,
        times,
        return_density_matrices=True,
        **options,
    )
    reference = _dense_trajectory(
        hamiltonian, n_wires, times, collapse_rate=collapse_rate
    )
    assert result.density_matrices is not None
    # Every step here is complex128 arithmetic over at most 4 wires, where the
    # two routes differ only by summation order; a sign or stride error in the
    # Pauli action moves the trajectory by O(1) instead.
    assert float((result.density_matrices - reference).abs().max()) < 1e-15


def test_the_observable_trajectory_matches_the_dense_expectation_values() -> None:
    hamiltonian = 0.5 * fq.X(0) + 0.3 * (fq.X(0) @ fq.X(1)) + 0.2 * (fq.Z(0) @ fq.Z(1))
    operators = {"z0": fq.Z(0), "x0x1": fq.X(0) @ fq.X(1)}
    result = evolve_density_matrix(
        hamiltonian,
        "00",
        2,
        [0.0, 0.1, 0.2],
        return_density_matrices=True,
        observables=operators,
    )
    reference = _dense_trajectory(hamiltonian, 2, [0.0, 0.1, 0.2])
    for name, operator in operators.items():
        expected = torch.real(
            torch.einsum(
                "tij,ji->t",
                reference,
                lindblad_module._observable_matrix(
                    operator,
                    n_wires=2,
                    dtype=torch.complex128,
                    device=_DEVICE,
                    field="observables",
                ),
            )
        )
        assert float((result.observables[name] - expected).abs().max()) < 1e-14


@pytest.mark.parametrize("block_bytes", [1, 64, 4096, 8 * 1024 * 1024])
def test_any_block_budget_produces_the_same_commutator(block_bytes: int) -> None:
    """The row blocking is a working-set bound, never a numerical choice."""

    hamiltonian = (
        0.5 * fq.X(0)
        - 0.25 * (fq.Y(0) @ fq.Y(1))
        + 0.75 * (fq.Z(1) @ fq.Z(2))
        + 1.5 * fq.Z(2)
    )
    representation = _normalize(hamiltonian, n_wires=3)
    assert isinstance(representation, PauliSum)
    state = torch.randn(8, 8, dtype=torch.complex128)
    reference = representation.commutator(state, block_bytes=8 * 1024 * 1024)
    blocked = representation.commutator(state, block_bytes=block_bytes)
    assert torch.equal(blocked, reference)


def test_the_block_budget_bounds_the_working_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget must change how many rows are held, or it bounds nothing.

    Every block size is numerically equivalent, so a wrong budget is invisible
    to a comparison of results; the number of blocks is what makes the bound
    observable. The temporaries per row are three complex128 rows of four
    entries each, 192 bytes at 2 wires, so a 192-byte budget holds exactly
    one row while a 768-byte budget holds all four. The complex64 phase repeats
    the measurement at half the width, where the same 192 bytes hold two rows:
    an estimate that assumed one element width would take four.
    """

    hamiltonian = 0.5 * fq.X(0) + 0.75 * (fq.Z(0) @ fq.Z(1))
    representation = _normalize(hamiltonian, n_wires=2)
    assert isinstance(representation, PauliSum)
    state = torch.randn(4, 4, dtype=torch.complex128)

    blocks: list[tuple[int, int]] = []
    original = PauliSum._commutator_block

    def recorded(
        self: PauliSum, value: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        blocks.append((start, stop))
        return original(self, value, start, stop)

    monkeypatch.setattr(PauliSum, "_commutator_block", recorded)
    representation.commutator(state, block_bytes=192)
    assert blocks == [(0, 1), (1, 2), (2, 3), (3, 4)]

    blocks.clear()
    representation.commutator(state, block_bytes=768)
    assert blocks == [(0, 4)]

    narrow = _normalize(hamiltonian, n_wires=2, dtype=torch.complex64)
    assert isinstance(narrow, PauliSum)
    blocks.clear()
    narrow.commutator(state.to(torch.complex64), block_bytes=192)
    assert blocks == [(0, 2), (2, 4)]


def test_the_matrix_products_are_the_two_one_sided_actions() -> None:
    """``H @ rho`` and ``rho @ H`` must not be interchangeable.

    Both actions are checked against the dense product, and against each other:
    for a Hermitian ``H`` the two agree only when ``rho`` commutes with ``H``, so
    the fixture uses a state that does not. A swapped dispatch would still match
    the dense form of the other side here and is therefore caught. Measured
    2.2e-16 and 4.4e-16 against the dense products, 4.4e+00 between the two
    sides, and exactly 0.0 between the commutator and their difference.
    """

    hamiltonian = 0.5 * fq.X(0) + 0.75 * (fq.Y(0) @ fq.Z(1)) + 1.25 * fq.Z(1)
    representation = _normalize(hamiltonian, n_wires=2)
    assert isinstance(representation, PauliSum)
    dense = representation.dense()
    state = torch.randn(4, 4, dtype=torch.complex128)

    left = representation @ state
    right = state @ representation
    assert torch.allclose(left, dense @ state, atol=1e-14)
    assert torch.allclose(right, state @ dense, atol=1e-14)
    assert float(torch.max(torch.abs(left - right))) > 1e-2
    # The commutator is the difference of the two one-sided results.
    assert torch.allclose(representation.commutator(state), left - right, atol=1e-14)


def test_the_matrix_products_carry_the_batch_axis() -> None:
    hamiltonian = 0.5 * fq.X(0) + 0.75 * (fq.Z(0) @ fq.Z(1))
    representation = _normalize(hamiltonian, n_wires=2)
    assert isinstance(representation, PauliSum)
    dense = representation.dense()
    batch = torch.randn(3, 4, 4, dtype=torch.complex128)

    assert torch.allclose(representation @ batch, dense @ batch, atol=1e-14)
    assert torch.allclose(batch @ representation, batch @ dense, atol=1e-14)


def test_the_matrix_products_share_the_commutator_block_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-sided product is blocked by the same rule, not left unbounded.

    The budget is read at call time, so it can be lowered to the 192 bytes that
    hold one complex128 row at 2 wires; a product that ignored it would show one
    block instead of four.
    """

    hamiltonian = 0.5 * fq.X(0) + 0.75 * (fq.Z(0) @ fq.Z(1))
    representation = _normalize(hamiltonian, n_wires=2)
    assert isinstance(representation, PauliSum)
    dense = representation.dense()
    state = torch.randn(4, 4, dtype=torch.complex128)

    blocks: list[tuple[int, int]] = []
    original = PauliSum._left_block

    def recorded(
        self: PauliSum, value: torch.Tensor, start: int, stop: int
    ) -> torch.Tensor:
        blocks.append((start, stop))
        return original(self, value, start, stop)

    monkeypatch.setattr(PauliSum, "_left_block", recorded)
    monkeypatch.setattr(
        "flagquantum.simulation.matrix_free_hamiltonian."
        "DEFAULT_COMMUTATOR_BLOCK_BYTES",
        192,
    )
    assert torch.allclose(representation @ state, dense @ state, atol=1e-14)
    assert blocks == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_the_matrix_products_refuse_a_state_the_commutator_refuses() -> None:
    representation = _normalize(1.5 * fq.Z(0), n_wires=1)
    assert isinstance(representation, PauliSum)
    with pytest.raises(ValueError, match="shape"):
        representation @ torch.zeros(3, 3, dtype=torch.complex128)
    with pytest.raises(ValueError, match="shape"):
        torch.zeros(3, 3, dtype=torch.complex128) @ representation
    with pytest.raises(ValueError, match="dtype and device"):
        representation @ torch.zeros(2, 2, dtype=torch.complex64)


def test_the_representation_is_not_a_matrix() -> None:
    """The plan must not pay ``4**n`` for a Hamiltonian it never applies densely."""

    hamiltonian = sum(
        (0.1 * (fq.Z(wire) @ fq.Z(wire + 1)) for wire in range(9)), fq.Z(0) * 0.0
    )
    representation = _normalize(hamiltonian, n_wires=12)
    assert isinstance(representation, PauliSum)
    assert representation.terms
    assert not isinstance(representation, torch.Tensor)
    assert all(isinstance(term, PauliSumTerm) for term in representation.terms)
    # The dense matrix at twelve wires is 4096**2 complex128 entries, 256 MiB.
    assert len(representation.terms) <= 12


def test_the_commutator_refuses_a_state_it_cannot_apply() -> None:
    representation = _normalize(1.5 * fq.Z(0), n_wires=1)
    assert isinstance(representation, PauliSum)
    with pytest.raises(ValueError, match="shape"):
        representation.commutator(torch.zeros(3, 3, dtype=torch.complex128))
    with pytest.raises(ValueError, match="dtype and device"):
        representation.commutator(torch.zeros(2, 2, dtype=torch.complex64))


def test_an_empty_pauli_sum_is_the_identity_free_zero_operator() -> None:
    representation = PauliSum(
        (),
        n_wires=1,
        dimension=2,
        dtype=torch.complex128,
        device=_DEVICE,
    )
    state = torch.randn(2, 2, dtype=torch.complex128)
    assert torch.equal(representation.commutator(state), torch.zeros_like(state))


@pytest.mark.parametrize(
    ("hamiltonian", "expected_code"),
    [
        (_complex_term(1j, ((0, "z"),)), "non_hermitian_hamiltonian"),
        ([{"pauli": "Z", "wires": [0], "coefficient": 1j}], "invalid_hamiltonian"),
        ([{"pauli": "XX", "wires": [0, 0], "coefficient": 1.0}], "invalid_operator"),
        ([{"pauli": "ZZ", "wires": [0, 0], "coefficient": 1.0}], "invalid_operator"),
        ([{"pauli": "Z", "wires": [5], "coefficient": 1.0}], "invalid_operator"),
        ([{"pauli": "Z", "wires": [-1], "coefficient": 1.0}], "invalid_operator"),
        ([{"pauli": "ZZ", "wires": [0], "coefficient": 1.0}], "invalid_operator"),
        ([{"pauli": "Q", "wires": [0], "coefficient": 1.0}], "unknown_operator"),
        (
            [{"pauli": "Z", "wires": [0], "coefficient": float("inf")}],
            "invalid_hamiltonian",
        ),
        ([{"pauli": "Z", "wires": [0]}, 3], "invalid_hamiltonian"),
    ],
)
def test_the_representation_keeps_the_structured_refusals(
    hamiltonian: object, expected_code: str
) -> None:
    payload = (
        Observable(terms=(hamiltonian,))
        if isinstance(hamiltonian, _PauliTerm)
        else hamiltonian
    )
    with pytest.raises(EvolutionValidationError) as error:
        _normalize(payload, n_wires=1)
    assert error.value.code == expected_code


def test_a_duplicated_wire_is_refused_rather_than_resolved_by_placement() -> None:
    """A repeated wire has no Pauli semantics, so it cannot be silently placed.

    The dense route let the second placement overwrite the first, which made
    ``XX`` on ``(0, 0)`` non-Hermitian and ``ZZ`` on ``(0, 0)`` Hermitian:
    both are artefacts of the placement, not the requested Hamiltonian.
    """

    for symbol in ("XX", "ZZ", "YY", "ZX"):
        with pytest.raises(EvolutionValidationError, match="same wire twice"):
            _normalize(
                [{"pauli": symbol, "wires": [0, 0], "coefficient": 1.0}], n_wires=1
            )


def test_the_hermiticity_boundary_matches_the_dense_comparison() -> None:
    """The bound is the same ``atol``, so the accepted set cannot widen.

    The imaginary part of a coefficient appears twice in ``H - H^dagger``, so
    the matrix-free rule decides on ``2 * |Im(c)|`` while the dense comparison
    it replaces decided on ``max|H - H^dagger| <= atol``. The two rules agree
    everywhere, but only the band ``tol / 2 < |Im(c)| <= tol`` can tell them
    apart: below the band both accept, above it both reject. A sample outside
    that band passes whichever rule is in force, so the points below straddle
    it and every verdict is checked against the dense comparison as well.
    """

    for imaginary, accepted in (
        (5e-11, True),
        (5.1e-11, False),
        (7e-11, False),
        (1e-10, False),
        (1.1e-10, False),
    ):
        hamiltonian = Observable(terms=(_PauliTerm(1.0 + 1j * imaginary, ((0, "z"),)),))
        reference = _dense_hamiltonian(hamiltonian, n_wires=1)
        dense_accepts = bool(
            torch.allclose(reference, reference.conj().T, atol=_TOLERANCE, rtol=0.0)
        )
        assert dense_accepts is accepted
        try:
            representation = _normalize(hamiltonian, n_wires=1)
        except EvolutionValidationError as error:
            assert error.code == "non_hermitian_hamiltonian"
            assert not accepted
        else:
            assert accepted
            assert isinstance(representation, PauliSum)
            assert torch.equal(representation.dense(), reference)


def test_a_real_coefficient_on_a_y_string_stays_hermitian() -> None:
    """``Y`` is Hermitian; only the action word needs the ``i``."""

    for factors in (
        ((0, "y"),),
        ((0, "y"), (1, "y")),
        ((0, "x"), (1, "y")),
        ((0, "y"), (1, "z")),
    ):
        hamiltonian = Observable(terms=(_PauliTerm(0.5, factors),))
        representation = _normalize(hamiltonian, n_wires=2)
        assert isinstance(representation, PauliSum)
        assert torch.equal(
            representation.dense(), _dense_hamiltonian(hamiltonian, n_wires=2)
        )


def _non_hermitian_pauli_sum() -> PauliSum:
    """Return a matrix-free sum whose coefficients are genuinely complex.

    The words overlap so that the ``i`` factors ``action_coefficient`` applies
    are exercised, and one word carries a ``Y``, which is the factor whose
    transpose differs from itself. A Hermitian sum would not separate the
    conjugation from the identity.
    """

    return PauliSum(
        (
            PauliSumTerm(complex(0.3, -0.7), mask=0b11, signs=(0,)),
            PauliSumTerm(complex(-0.25, 0.4), mask=0b1, signs=(1,)),
            PauliSumTerm(complex(0.8, 0.15), mask=0b10, signs=(0, 1)),
        ),
        n_wires=2,
        dimension=4,
        dtype=torch.complex128,
        device=_DEVICE,
    )


def test_the_adjoint_of_a_pauli_sum_is_its_conjugate_transpose() -> None:
    representation = _non_hermitian_pauli_sum()
    adjoint = representation.adjoint()
    dense = representation.dense()

    assert adjoint.dimension == representation.dimension
    assert adjoint.dtype == representation.dtype
    assert len(adjoint.terms) == len(representation.terms)
    # Two routes to the same matrix: ``dense`` sums the action words column by
    # column, ``adjoint`` rebuilds the terms. Measured worst entry 0.0.
    assert float((adjoint.dense() - dense.conj().T).abs().max()) == 0.0
    # Controls: the plain transpose and the sum itself are different operators,
    # so the agreement above is about the conjugation and not about symmetry.
    assert float((adjoint.dense() - dense.T).abs().max()) > 1e-2
    assert float((adjoint.dense() - dense).abs().max()) > 1e-2


def test_the_matrix_free_adjoint_pairs_with_the_matrix_free_action() -> None:
    """``<sigma, H rho> = <H^dag sigma, rho>`` through the blocked actions only.

    Both sides go through ``__matmul__``/``__rmatmul__``, the per-block kernels
    that never build ``H``, so this is evidence about the adjoint of the action
    and not a restatement of the dense view.
    """

    representation = _non_hermitian_pauli_sum()
    adjoint = representation.adjoint()
    generator = torch.Generator().manual_seed(31)
    rho = torch.randn(4, 4, dtype=torch.complex128, generator=generator)
    sigma = torch.randn(4, 4, dtype=torch.complex128, generator=generator)
    inner = lambda left, right: (left.conj().T @ right).trace()  # noqa: E731

    forward = inner(sigma, representation @ rho)
    backward = inner(adjoint @ sigma, rho)

    assert float(abs(forward)) > 0.5
    assert float(abs(forward - backward)) < 1e-14
    # Control: the sum used as its own adjoint -- the rule that drops the
    # conjugation -- does not pair. Measured residual 3.83e0 against this floor.
    assert float(abs(inner(representation @ sigma, rho) - forward)) > 1e-2


def test_a_hermitian_pauli_sum_is_its_own_adjoint() -> None:
    """The invariant that lets a planned Hamiltonian be adjointed at all."""

    representation = _normalize(0.5 * fq.X(0) + 0.25 * (fq.Z(0) @ fq.Z(1)), n_wires=2)
    assert isinstance(representation, PauliSum)

    adjoint = representation.adjoint()

    # Conjugating a real coefficient is a value equality, not an identity, so
    # this is a statement about the term list the adjoint rebuilt.
    assert adjoint == representation
    assert adjoint is not representation
    assert float((adjoint.dense() - representation.dense()).abs().max()) == 0.0


def test_adjointing_a_pauli_sum_does_not_materialize_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adjoint is a term-list operation, so it must not pay ``4**n``."""

    representation = _non_hermitian_pauli_sum()

    def forbidden(self: PauliSum) -> torch.Tensor:
        raise AssertionError("adjointing a PauliSum materialized the Hamiltonian")

    monkeypatch.setattr(PauliSum, "dense", forbidden)
    adjoint = representation.adjoint()
    state = torch.randn(4, 4, dtype=torch.complex128)

    # Both directions of the adjointed factor are exercised, because a
    # superoperator term can place the factor on either side of the state.
    mapped = SuperOperator.left_multiply(representation).adjoint()
    assert torch.equal(mapped.apply(state), adjoint @ state)
    assert torch.equal(
        SuperOperator.right_multiply(representation).adjoint().apply(state),
        state @ adjoint,
    )


def test_planning_does_not_materialize_the_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning decides Hermiticity without building ``4**n`` entries.

    The dense matrix is still what the serialized request stores, so this
    records the one remaining allocation instead of hiding it.
    """

    hamiltonian = 0.5 * fq.X(0) + 0.25 * (fq.Z(0) @ fq.Z(1))
    plan = plan_density_matrix_evolution(hamiltonian, "00", 2, [0.0, 0.1])
    assert plan.trajectory_bytes > 0

    def forbidden(self: PauliSum) -> torch.Tensor:
        raise AssertionError("evolution planning materialized the Hamiltonian")

    monkeypatch.setattr(PauliSum, "dense", forbidden)
    plan_density_matrix_evolution(hamiltonian, "00", 2, [0.0, 0.1])


def test_the_serialized_request_schema_is_unchanged() -> None:
    """The plan payload keeps its shape, so stored plans stay readable."""

    from flagquantum.lindblad import plan as build_plan

    hamiltonian = 0.5 * fq.X(0) + 0.25 * fq.Z(0)
    plan = build_plan(hamiltonian, "0", [0.0, 0.1], n_qubits=1)
    payload = plan.to_dict()
    assert payload["schema"] == "flagquantum.lindblad_plan"
    assert set(payload["request"]["hamiltonian"]) == {"shape", "values"}
    assert payload["request"]["hamiltonian"]["shape"] == [2, 2]

    restored = type(plan).from_json(plan.to_json())
    assert restored.to_dict() == payload

    tampered = plan.to_dict()
    tampered["request"]["times"][1] = 0.2
    with pytest.raises(SerializationError, match="identity"):
        type(plan).from_dict(tampered)


def test_the_evolving_hamiltonian_is_a_pauli_sum_for_every_accepted_input() -> None:
    """Every Pauli input form reaches the matrix-free path, matrices do not."""

    for hamiltonian in (
        0.5 * fq.X(0) + 0.25 * fq.Z(0),
        [{"pauli": "X", "wires": [0], "coefficient": 0.5}],
    ):
        assert isinstance(_normalize(hamiltonian, n_wires=1), PauliSum)

    dense = torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=torch.complex128)
    assert isinstance(_normalize(dense, n_wires=1), torch.Tensor)


def test_a_pauli_sum_handed_back_is_evolved_without_densifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The representation is an accepted Hamiltonian form, not only an output.

    A caller holding a representation can hand it straight back, so all three
    Pauli forms must reach the same trajectory -- and the form that is already
    matrix-free must not be read out as a matrix on the way. The three routes
    are bitwise equal here, which is why the comparison can be ``torch.equal``
    rather than a tolerance: they differ only in how the terms were named.
    """

    representation = PauliSum(
        (
            PauliSumTerm(coefficient=0.5, mask=0b1, signs=()),
            PauliSumTerm(coefficient=0.25, mask=0, signs=(0,)),
        ),
        n_wires=1,
        dimension=2,
        dtype=torch.complex128,
        device=_DEVICE,
    )
    observable = 0.5 * fq.X(0) + 0.25 * fq.Z(0)
    descriptors = [
        {"pauli": "X", "wires": [0], "coefficient": 0.5},
        {"pauli": "Z", "wires": [0], "coefficient": 0.25},
    ]
    times = [0.0, 0.05, 0.11, 0.2]
    expected = evolve_density_matrix(
        observable, "0", 1, times, return_density_matrices=True
    )

    def forbidden(self: PauliSum) -> torch.Tensor:
        raise AssertionError("evolving a PauliSum materialized the Hamiltonian")

    monkeypatch.setattr(PauliSum, "dense", forbidden)
    for candidate in (representation, descriptors):
        result = evolve_density_matrix(
            candidate, "0", 1, times, return_density_matrices=True
        )
        assert result.density_matrices is not None
        assert torch.equal(result.density_matrices, expected.density_matrices)
