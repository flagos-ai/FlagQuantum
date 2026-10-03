"""Superoperator algebra: constructors, both views, and their cross-checks.

Every dense expectation here is built by an oracle written independently of the
implementation. ``_matrix_from_action`` defines the row-major vectorization by
its *action* -- it applies the map to each basis matrix ``E_ij`` and stacks the
flattened results as columns -- so it cannot inherit a convention error from a
Kronecker product. ``_kronecker_sum`` writes the explicit ``A (x) B^T`` form, so
a disagreement between the two oracles would be a real mathematical conflict
rather than a shared assumption.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.operators import DEFAULT_DENSE_MATRIX_BYTES, SuperOperator
from flagquantum.simulation.lindblad_generator import Liouvillian

pytestmark = pytest.mark.unit

_DIMENSION = 3
_COMPLEX128 = torch.complex128
_COMPLEX64 = torch.complex64


def _complex_factors(dimension: int = _DIMENSION) -> tuple[torch.Tensor, ...]:
    """Return complex, non-normal factors, so conjugation is distinguishable."""

    generator = torch.Generator().manual_seed(20250930)
    factors = []
    for _ in range(4):
        real = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
        imaginary = torch.randn(
            dimension, dimension, dtype=torch.float64, generator=generator
        )
        factors.append((real + 1j * imaginary).to(_COMPLEX128))
    return tuple(factors)


def _density_matrix(dimension: int = _DIMENSION) -> torch.Tensor:
    """Return a complex, non-diagonal, unit-trace density matrix."""

    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(dimension, dimension, dtype=torch.complex128, generator=generator)
    state = raw @ raw.conj().T
    return state / state.trace()


def _matrix_from_action(
    terms: tuple[tuple[complex, torch.Tensor | None, torch.Tensor | None], ...],
    dimension: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the row-major matrix of the map from the map's own action."""

    columns = []
    for row in range(dimension):
        for column in range(dimension):
            basis = torch.zeros(dimension, dimension, dtype=dtype)
            basis[row, column] = 1.0 + 0.0j
            value = torch.zeros(dimension, dimension, dtype=dtype)
            for coefficient, left, right in terms:
                term = basis
                if left is not None:
                    term = left @ term
                if right is not None:
                    term = term @ right
                value = value + coefficient * term
            columns.append(value.reshape(-1))
    return torch.stack(columns, dim=1)


def _kronecker_sum(
    terms: tuple[tuple[complex, torch.Tensor | None, torch.Tensor | None], ...],
    dimension: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Write the explicit ``vec(A rho B) = (A (x) B^T) vec(rho)`` form."""

    identity = torch.eye(dimension, dtype=dtype)
    total = torch.zeros(dimension**2, dimension**2, dtype=dtype)
    for coefficient, left, right in terms:
        if left is None:
            block = torch.kron(identity, right.transpose(0, 1).contiguous())
        elif right is None:
            block = torch.kron(left, identity)
        else:
            block = torch.kron(left, right.transpose(0, 1).contiguous())
        total = total + coefficient * block
    return total


def _dissipator(
    lowering: torch.Tensor,
) -> tuple[SuperOperator, tuple[tuple[complex, torch.Tensor | None, torch.Tensor | None], ...]]:
    """Return the Lindblad dissipator as a map and as its plain term tuple."""

    number = lowering.conj().T @ lowering
    applied = SuperOperator.left_right_multiply(lowering, lowering.conj().T)
    applied += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
    applied += SuperOperator.right_multiply((-0.5 + 0.0j) * number)
    plain = (
        (1.0 + 0.0j, lowering, lowering.conj().T),
        (-0.5 + 0.0j, number, None),
        (-0.5 + 0.0j, None, number),
    )
    return applied, plain


def test_the_package_reexports_the_operator_schema_and_the_superoperator() -> None:
    import flagquantum.operators as operators

    # The two names the shim this package replaces already exported must keep
    # working by the same import line, because two test modules use them.
    assert operators.gate_info("x").name == "x"
    assert operators.GateInfo is not None
    assert operators.SuperOperator is SuperOperator
    assert operators.DEFAULT_DENSE_MATRIX_BYTES == DEFAULT_DENSE_MATRIX_BYTES


def test_the_three_constructors_record_one_term_apiece() -> None:
    left, right = _complex_factors()[:2]

    assert len(SuperOperator.left_multiply(left)) == 1
    assert SuperOperator.left_multiply(left).terms == ((1.0 + 0.0j, left, None),)
    assert SuperOperator.right_multiply(right).terms == ((1.0 + 0.0j, None, right),)
    assert SuperOperator.left_right_multiply(left, right).terms == (
        (1.0 + 0.0j, left, right),
    )


def test_an_empty_superoperator_refuses_to_report_a_dimension_or_a_dtype() -> None:
    empty = SuperOperator()

    assert len(empty) == 0
    assert list(empty) == []
    with pytest.raises(ValueError, match="an empty superoperator has no dimension"):
        _ = empty.dimension
    with pytest.raises(ValueError, match="an empty superoperator has no dimension"):
        _ = empty.dtype
    with pytest.raises(ValueError, match="an empty superoperator has no dimension"):
        empty.dense()
    with pytest.raises(ValueError, match="an empty superoperator has no dimension"):
        empty.apply(torch.zeros(_DIMENSION, _DIMENSION, dtype=_COMPLEX128))


def test_the_empty_instance_is_an_accumulator() -> None:
    left, right = _complex_factors()[:2]
    accumulator = SuperOperator()

    accumulator += SuperOperator.left_right_multiply(left, right)

    assert accumulator.dimension == _DIMENSION
    assert accumulator.dtype == _COMPLEX128
    assert len(accumulator) == 1
    empty_other = SuperOperator()
    accumulator += empty_other
    assert len(accumulator) == 1


def test_the_dense_form_equals_the_matrix_built_from_the_action() -> None:
    left, right, extra, second = _complex_factors()
    applied = SuperOperator.left_right_multiply(left, right)
    applied += SuperOperator.left_multiply(extra)
    applied += SuperOperator.right_multiply(0.25 * second)

    measured = applied.dense()
    oracle = _matrix_from_action(applied.terms, _DIMENSION, _COMPLEX128)

    assert measured.shape == (_DIMENSION**2, _DIMENSION**2)
    # Both sides build the same matrix by different routes, so the agreement is
    # at the round-off floor rather than at a chosen tolerance.
    assert float(torch.max(torch.abs(measured - oracle))) < 1e-15
    assert torch.allclose(measured, oracle, atol=1e-15)


def test_the_dense_form_equals_the_explicit_kronecker_sum() -> None:
    applied, plain = _dissipator(_complex_factors()[0])

    measured = applied.dense()
    oracle = _kronecker_sum(plain, _DIMENSION, _COMPLEX128)

    # Measured worst-entry difference 0.0: the two sums add the same five blocks
    # in the same order at the same precision.
    assert float(torch.max(torch.abs(measured - oracle))) == 0.0


def test_the_right_factor_is_transposed_and_not_conjugate_transposed() -> None:
    left, right = _complex_factors()[:2]
    applied = SuperOperator.left_right_multiply(left, right)
    identity = torch.eye(_DIMENSION, dtype=_COMPLEX128)
    state = _density_matrix()
    expected = (left @ state @ right).reshape(-1)

    correct = applied.dense() @ state.reshape(-1)
    transposed_wrong = torch.kron(left, right.conj()) @ state.reshape(-1)
    commutator_shape_wrong = torch.kron(right.transpose(0, 1) @ left, identity) @ (
        state.reshape(-1)
    )
    swapped_wrong = torch.kron(right.transpose(0, 1).contiguous(), left) @ (
        state.reshape(-1)
    )

    assert float(torch.max(torch.abs(correct - expected))) < 1e-15
    # Each wrong rule must miss the tolerance by orders of magnitude, or the
    # assertion above is not evidence of the convention.
    assert float(torch.max(torch.abs(transposed_wrong - expected))) > 1e-2
    assert float(torch.max(torch.abs(commutator_shape_wrong - expected))) > 1e-2
    assert float(torch.max(torch.abs(swapped_wrong - expected))) > 1e-2


def test_a_one_sided_term_reduces_to_the_identity_kronecker_form() -> None:
    left, _, _, right = _complex_factors()
    state = _density_matrix()
    identity = torch.eye(_DIMENSION, dtype=_COMPLEX128)

    left_only = SuperOperator.left_multiply(left)
    right_only = SuperOperator.right_multiply(right)

    assert torch.equal(left_only.dense(), torch.kron(left, identity))
    assert torch.equal(right_only.dense(), torch.kron(identity, right.T.contiguous()))
    # The dense form and the action agree for one-sided terms too, and the
    # identity factor is what distinguishes the two sides.
    assert float(
        torch.max(
            torch.abs(left_only.dense() @ state.reshape(-1) - (left @ state).reshape(-1))
        )
    ) < 1e-15
    assert float(
        torch.max(
            torch.abs(
                right_only.dense() @ state.reshape(-1) - (state @ right).reshape(-1)
            )
        )
    ) < 1e-15
    assert not torch.allclose(
        left_only.dense(), torch.kron(identity, left.T.contiguous()), atol=1e-3
    )


def test_the_dense_form_vectorises_the_action_on_a_density_matrix() -> None:
    applied, _ = _dissipator(_complex_factors()[0])
    state = _density_matrix()

    from_dense = (applied.dense() @ state.reshape(-1)).reshape(_DIMENSION, _DIMENSION)
    from_action = applied.apply(state)

    # The two views are independent formulas, so this cross-check is evidence for
    # both rather than for one derived from the other.
    assert float(torch.max(torch.abs(from_dense - from_action))) < 1e-15
    assert torch.allclose(from_dense, from_action, atol=1e-14)


def test_apply_preserves_a_batch_axis() -> None:
    applied, _ = _dissipator(_complex_factors()[0])
    first = _density_matrix()
    second = 0.5 * (torch.eye(_DIMENSION, dtype=_COMPLEX128) + first)
    batch = torch.stack([first, second], dim=0)

    batched = applied.apply(batch)

    assert batched.shape == (2, _DIMENSION, _DIMENSION)
    for index in range(2):
        assert torch.allclose(batched[index], applied.apply(batch[index]), atol=1e-15)


def test_the_lindblad_dissipator_is_trace_preserving() -> None:
    applied, _ = _dissipator(_complex_factors()[0])
    identity = torch.eye(_DIMENSION, dtype=_COMPLEX128)

    # Trace preservation is L^T vec(I) = 0 in the row-major vectorization.
    from_dense = applied.dense().T @ identity.reshape(-1)
    from_action = applied.apply(identity).reshape(-1)

    assert float(torch.linalg.norm(from_dense)) < 1e-15
    assert float(torch.linalg.norm(from_action)) < 1e-15
    state = _density_matrix()
    trace = applied.apply(state).trace()
    assert abs(complex(trace)) < 1e-15


def test_the_lindblad_dissipator_matches_the_liouvillian_generator() -> None:
    lowering = _complex_factors()[0]
    generator = Liouvillian(
        _hamiltonian(), [lowering], hilbert_dimension=_DIMENSION
    )
    number = lowering.conj().T @ lowering
    applied = SuperOperator.left_multiply((-1.0j) * _hamiltonian())
    applied += SuperOperator.right_multiply(1.0j * _hamiltonian())
    applied += SuperOperator.left_right_multiply(lowering, lowering.conj().T)
    applied += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
    applied += SuperOperator.right_multiply((-0.5 + 0.0j) * number)

    measured = float(torch.max(torch.abs(applied.dense() - generator.dense())))
    # Measured 0.0: the two paths build the same five blocks and add them in the
    # same order, so the agreement is bitwise. Dropping the transpose on either
    # right factor moves this past 1e-1.
    assert measured < 1e-15
    assert torch.allclose(applied.dense(), generator.dense(), atol=1e-15)
    # The action agrees too, on a state that is not the identity, so the anchor
    # is not an artifact of a trace-preserving cancellation.
    state = _density_matrix()
    assert torch.allclose(
        applied.apply(state), generator.derivative(state), atol=1e-14
    )


def _hamiltonian() -> torch.Tensor:
    """Return a Hermitian, complex, non-diagonal 3x3 Hamiltonian."""

    generator = torch.Generator().manual_seed(11)
    real = torch.randn(_DIMENSION, _DIMENSION, dtype=torch.float64, generator=generator)
    imaginary = torch.randn(
        _DIMENSION, _DIMENSION, dtype=torch.float64, generator=generator
    )
    raw = (real + 1j * imaginary).to(_COMPLEX128)
    return 0.5 * (raw + raw.conj().T)


def test_scalar_multiplication_scales_every_coefficient() -> None:
    applied, _ = _dissipator(_complex_factors()[0])

    scaled = 2.0j * applied
    reflected = applied * 2.0j

    assert len(scaled) == len(applied)
    assert scaled == reflected
    assert scaled != applied
    for original, doubled in zip(applied.terms, scaled.terms, strict=True):
        assert doubled[0] == 2.0j * original[0]
    # The factors are untouched, so a matrix-free factor needs no in-place scale.
    assert scaled.terms[0][1] is applied.terms[0][1]
    assert torch.allclose(
        scaled.dense(), 2.0j * applied.dense(), atol=1e-15
    )
    with pytest.raises(ValueError, match="coefficient must be a scalar"):
        applied * "two"
    with pytest.raises(ValueError, match="coefficient must be a scalar"):
        applied * True


def test_addition_concatenates_terms_and_does_not_mutate_its_operands() -> None:
    left, right = _complex_factors()[:2]
    first = SuperOperator.left_multiply(left)
    second = SuperOperator.right_multiply(right)

    total = first + second

    assert len(total) == 2
    assert len(first) == 1 and len(second) == 1
    assert total.terms == first.terms + second.terms
    assert first == SuperOperator.left_multiply(left)


def test_equality_is_structural_and_order_sensitive() -> None:
    left, right = _complex_factors()[:2]
    first = SuperOperator.left_multiply(left)
    second = SuperOperator.right_multiply(right)

    assert first == SuperOperator.left_multiply(left)
    assert first == SuperOperator.left_multiply(left.clone())
    assert first != second
    # Concatenation order is observable, exactly as CUDA-Q's term-preserving
    # ``+=`` makes it observable, so equality does not canonicalize.
    assert (first + second) != (second + first)
    assert (first + second) == (first + second)


def test_equality_with_a_foreign_object_is_false_and_instances_are_unhashable() -> None:
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    assert (applied == "not a superoperator") is False
    assert (applied != "not a superoperator") is True
    with pytest.raises(TypeError, match="unhashable"):
        hash(applied)


def test_mismatched_dimensions_and_dtypes_are_refused() -> None:
    second_dimension = SuperOperator.left_multiply(
        torch.eye(2, dtype=_COMPLEX128)
    ).terms[0][1]
    wide = SuperOperator.left_multiply(torch.eye(_DIMENSION, dtype=_COMPLEX128))
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    with pytest.raises(ValueError, match="must share one dimension"):
        SuperOperator.left_right_multiply(_complex_factors()[0], second_dimension)
    with pytest.raises(ValueError, match="must share one dimension"):
        wide + applied
    with pytest.raises(ValueError, match="must share one dtype"):
        SuperOperator.left_multiply(torch.eye(2, dtype=_COMPLEX64)) + (
            SuperOperator.left_multiply(torch.eye(2, dtype=_COMPLEX128))
        )


def test_a_failed_in_place_addition_leaves_the_accumulator_unchanged() -> None:
    applied, _ = _dissipator(_complex_factors()[0])
    before = applied.terms
    foreign = SuperOperator.left_multiply(torch.eye(2, dtype=_COMPLEX128))

    with pytest.raises(ValueError, match="must share one dimension"):
        applied += foreign

    assert applied.terms == before
    assert applied.dimension == _DIMENSION


def test_a_non_square_or_non_complex_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a square matrix"):
        SuperOperator.left_multiply(torch.zeros(2, 3, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must have a complex dtype"):
        SuperOperator.left_multiply(torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="must have a complex dtype"):
        SuperOperator.left_multiply(torch.eye(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="must expose a torch dtype"):
        SuperOperator.left_multiply(object())


def test_applying_a_state_of_the_wrong_shape_or_dtype_is_refused() -> None:
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    with pytest.raises(ValueError, match="must match the superoperator dtype"):
        applied.apply(torch.zeros(_DIMENSION, _DIMENSION, dtype=_COMPLEX64))
    with pytest.raises(ValueError, match="must have shape"):
        applied.apply(torch.zeros(2, 2, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must have shape"):
        applied.apply(torch.zeros(_DIMENSION, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must return a tensor"):
        applied.apply([1.0, 0.0])


def test_the_dense_form_is_refused_above_its_byte_ceiling() -> None:
    four = SuperOperator.left_multiply(torch.eye(4, dtype=_COMPLEX128))
    # 4**4 = 256 entries of 16 bytes is exactly 4096 bytes.
    assert four.dense(max_bytes=4096).numel() * 16 == 4096
    with pytest.raises(ValueError) as refused:
        four.dense(max_bytes=4095)
    message = str(refused.value)
    assert "16 x 16" in message
    assert "256 entries" in message
    assert "4096 bytes" in message
    assert "apply the map instead of materializing it" in message

    # The default ceiling is the same one the vectorized generator uses: it
    # admits dimension 32 and refuses dimension 64 without allocating.
    assert DEFAULT_DENSE_MATRIX_BYTES == 16 * 1024 * 1024
    large = SuperOperator.left_multiply(torch.eye(64, dtype=_COMPLEX128))
    with pytest.raises(ValueError) as refused_large:
        large.dense()
    large_message = str(refused_large.value)
    assert "4096 x 4096" in large_message
    assert "268435456 bytes" in large_message
    assert "16777216-byte ceiling" in large_message
