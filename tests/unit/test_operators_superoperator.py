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
        real = torch.randn(
            dimension, dimension, dtype=torch.float64, generator=generator
        )
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


def _hamiltonian(dimension: int = _DIMENSION) -> torch.Tensor:
    """Return a Hermitian, complex, non-diagonal Hamiltonian."""

    generator = torch.Generator().manual_seed(11)
    real = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    imaginary = torch.randn(
        dimension, dimension, dtype=torch.float64, generator=generator
    )
    raw = (real + 1j * imaginary).to(_COMPLEX128)
    return 0.5 * (raw + raw.conj().T)


def _dissipator(
    lowering: torch.Tensor,
) -> tuple[
    SuperOperator, tuple[tuple[complex, torch.Tensor | None, torch.Tensor | None], ...]
]:
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
    assert set(operators.__all__) == {
        "DEFAULT_DENSE_MATRIX_BYTES",
        "GateInfo",
        "SuperOperator",
        "gate_info",
    }
    for name in operators.__all__:
        assert getattr(operators, name) is not None


def test_the_repr_names_the_term_count_the_dimension_and_the_dtype() -> None:
    applied, _ = _dissipator(_complex_factors()[0])

    text = repr(applied)

    assert text.startswith("SuperOperator(terms=3")
    assert "dimension=3" in text
    assert str(_COMPLEX128) in text
    assert repr(SuperOperator()) == (
        "SuperOperator(terms=0, dimension=None, dtype=None)"
    )


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

    # Measured 4.578e-16 for the correct rule, against 4.914e+00 for dropping
    # the transpose, 4.664e+00 for multiplying the right factor's transpose on
    # the left instead, and 4.475e+00 for swapping the Kronecker operands. A
    # wrong rule that stayed near the tolerance would not be evidence.
    assert float(torch.max(torch.abs(correct - expected))) < 1e-15
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
    assert (
        float(
            torch.max(
                torch.abs(
                    left_only.dense() @ state.reshape(-1) - (left @ state).reshape(-1)
                )
            )
        )
        < 1e-15
    )
    assert (
        float(
            torch.max(
                torch.abs(
                    right_only.dense() @ state.reshape(-1) - (state @ right).reshape(-1)
                )
            )
        )
        < 1e-15
    )
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
    generator = applied.dense()

    # Trace preservation is the dual statement L^T vec(I) = 0 in the row-major
    # vectorization, i.e. the map is trace-annihilating on any state.
    dual = generator.T @ identity.reshape(-1)
    state = _density_matrix()

    # Measured 5.5e-17 relative to the generator norm of 2.186e+01; the same
    # residual without the two anticommutator terms is 7.6e-01, so the relative
    # form -- not an absolute floor -- is what makes this an assertion.
    assert float(torch.linalg.norm(dual)) / float(torch.linalg.norm(generator)) < 1e-15
    assert abs(complex(applied.apply(state).trace())) < 1e-15
    # Applying the map to the identity is NOT zero, and must not be: for a
    # non-normal L the dissipator sends I to L L^dag - L^dag L, while a trace
    # preserving map is only required to annihilate the trace.
    assert float(torch.linalg.norm(applied.apply(identity))) > 1.0


def test_the_lindblad_dissipator_matches_the_liouvillian_generator() -> None:
    lowering = _complex_factors()[0]
    generator = Liouvillian(_hamiltonian(), [lowering], hilbert_dimension=_DIMENSION)
    number = lowering.conj().T @ lowering
    applied = SuperOperator.left_multiply((-1.0j) * _hamiltonian())
    applied += SuperOperator.right_multiply(1.0j * _hamiltonian())
    applied += SuperOperator.left_right_multiply(lowering, lowering.conj().T)
    applied += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
    applied += SuperOperator.right_multiply((-0.5 + 0.0j) * number)

    measured = float(torch.max(torch.abs(applied.dense() - generator.dense())))
    # Measured exactly 0.0. It was 4.441e-16 while Liouvillian.dense() assembled
    # the same matrix through its own kron expression, so the value is the
    # observable consequence of routing the Lindblad generator through this
    # class: the two forms are now one computation rather than two that agree.
    # Losing that independence is why tests/unit/test_lindblad_vectorized_generator.py
    # keeps its own from-scratch dense reference; this assertion is what proves
    # the replacement happened at all, and the wrong-rule controls below are what
    # keep it from being vacuous.
    assert measured == 0.0
    assert torch.allclose(applied.dense(), generator.dense(), atol=1e-15)
    # A dense() that returned zeros would satisfy the equality above. Measured
    # 6.635712885219092 for this fixture.
    assert float(torch.max(torch.abs(generator.dense()))) > 1.0

    def _conjugated_right(operator: torch.Tensor) -> torch.Tensor:
        return operator.conj()

    def _unconjugated_right(operator: torch.Tensor) -> torch.Tensor:
        return operator.T.contiguous()

    # The hand-built sum above would agree with the generator even if both
    # dropped the transpose, so each wrong rule is substituted into the same sum
    # and must move the result. Measured 4.094e+00 for conjugate-instead-of-
    # transpose and 4.299e+00 for transpose-of-the-other-side.
    for wrong_rule in (_conjugated_right, _unconjugated_right):
        substituted = SuperOperator.left_multiply((-1.0j) * _hamiltonian())
        substituted += SuperOperator.right_multiply(1.0j * _hamiltonian())
        substituted += SuperOperator.left_right_multiply(
            lowering, wrong_rule(lowering.conj().T)
        )
        substituted += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
        substituted += SuperOperator.right_multiply((-0.5 + 0.0j) * number)
        assert (
            float(torch.max(torch.abs(substituted.dense() - generator.dense()))) > 1e-2
        )
    # The action agrees too, on a state that is not the identity, so the anchor
    # is not an artifact of a trace-preserving cancellation. Measured 0.0.
    state = _density_matrix()
    assert torch.allclose(applied.apply(state), generator.derivative(state), atol=1e-14)


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
    assert torch.allclose(scaled.dense(), 2.0j * applied.dense(), atol=1e-15)
    # The action must scale too. Every other fixture bakes its scalar into the
    # factor, so this is the only place a term coefficient is not one, and
    # without it a coefficient-dropping action would be indistinguishable.
    assert torch.allclose(
        scaled.apply(_density_matrix()), 2.0j * applied.apply(_density_matrix())
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


def test_iteration_yields_the_insertion_order() -> None:
    left, right, third = _complex_factors()[:3]
    mixed = SuperOperator.left_multiply(left)
    mixed += SuperOperator.right_multiply(right)
    mixed += SuperOperator.left_right_multiply(third, left)

    assert list(mixed) == list(mixed.terms)
    # Every constructor supplies the coefficient 1 and puts a scaled operator
    # into the factor, which is what CUDA-Q's constructors do too, so order is
    # pinned here by factor identity rather than by a coefficient value.
    assert [term[0] for term in mixed] == [1.0 + 0.0j] * 3
    assert [term[1] is None for term in mixed] == [False, True, False]
    assert [term[2] is None for term in mixed] == [True, False, False]
    assert mixed.terms[0][1] is left
    assert mixed.terms[1][2] is right
    assert mixed.terms[2][1] is third
    assert mixed.terms[2][2] is left


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
    # Each of the three compared components must discriminate on its own, or the
    # comparison is narrower than it claims to be. torch.equal is dtype
    # insensitive, so the dtype comparison is the only thing separating these.
    narrow = torch.eye(2, dtype=_COMPLEX64)
    wide = torch.eye(2, dtype=_COMPLEX128)
    assert torch.equal(narrow, wide)
    assert SuperOperator.left_multiply(narrow) != SuperOperator.left_multiply(wide)
    assert SuperOperator.left_multiply(left) != SuperOperator.left_multiply(right)
    assert SuperOperator.right_multiply(left) != SuperOperator.right_multiply(right)
    assert SuperOperator.left_multiply(left) != SuperOperator.right_multiply(left)
    assert SuperOperator.left_right_multiply(left, right) != (
        SuperOperator.left_right_multiply(right, left)
    )
    # A different term count must be reported as inequality, not as an error.
    assert first != first + first


def test_a_foreign_operand_is_refused_by_the_arithmetic_protocol() -> None:
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    # Returning NotImplemented is what lets Python try the reflected operation
    # and then raise TypeError, rather than silently succeeding.
    assert applied.__add__("not a superoperator") is NotImplemented
    assert applied.__iadd__("not a superoperator") is NotImplemented
    with pytest.raises(TypeError):
        applied + "not a superoperator"
    with pytest.raises(TypeError):
        applied += "not a superoperator"
    assert len(applied) == 1


def test_equality_with_a_foreign_object_is_false_and_instances_are_unhashable() -> None:
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    assert (applied == "not a superoperator") is False
    assert (applied != "not a superoperator") is True
    with pytest.raises(TypeError, match="unhashable"):
        hash(applied)


def test_mismatched_dimensions_and_dtypes_are_refused() -> None:
    narrow = torch.eye(2, dtype=_COMPLEX128)
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    with pytest.raises(ValueError, match="must share one dimension"):
        SuperOperator.left_right_multiply(_complex_factors()[0], narrow)
    with pytest.raises(ValueError, match="must share one dimension"):
        applied + SuperOperator.left_multiply(narrow)
    with pytest.raises(ValueError, match="must share one dimension"):
        SuperOperator.left_multiply(narrow) + applied
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


class _MatrixFactor:
    """A non-tensor matrix factor, used to exercise the protocol path.

    The matrix is real and asymmetric, so multiplying it on the left and on the
    right give different results and a dispatch in the wrong direction is
    visible. The two dunders are counted, and equality is by value.
    """

    def __init__(self, matrix: torch.Tensor, tag: str = "factor") -> None:
        self._matrix = matrix.to(torch.float64)
        self._tag = tag
        self.dimension = int(matrix.shape[0])
        self.dtype = _COMPLEX128
        self.matmul_calls = 0
        self.rmatmul_calls = 0

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor:
        self.matmul_calls += 1
        return self._matrix.to(state.dtype) @ state

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor:
        self.rmatmul_calls += 1
        return state @ self._matrix.to(state.dtype)

    def dense(self) -> torch.Tensor:
        return self._matrix.to(_COMPLEX128)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MatrixFactor):
            return NotImplemented
        return self._tag == other._tag and torch.equal(self._matrix, other._matrix)


def test_a_non_tensor_factor_is_dispatched_through_the_matching_dunder() -> None:
    matrix = torch.arange(9, dtype=torch.float64).reshape(_DIMENSION, _DIMENSION)
    square = torch.eye(_DIMENSION, dtype=_COMPLEX128)
    state = _density_matrix()
    left_factor = _MatrixFactor(matrix, tag="left")
    right_factor = _MatrixFactor(matrix, tag="right")

    mapped_left = SuperOperator.left_multiply(left_factor)
    mapped_right = SuperOperator.right_multiply(right_factor)

    # The matrix is asymmetric, so each direction is a distinct value and the
    # call counters say which dunder was reached.
    complex_matrix = matrix.to(_COMPLEX128)
    assert not torch.allclose(complex_matrix @ state, state @ complex_matrix)
    assert torch.equal(mapped_left.apply(state), complex_matrix @ state)
    assert torch.equal(mapped_right.apply(state), state @ complex_matrix)
    assert (left_factor.matmul_calls, left_factor.rmatmul_calls) == (1, 0)
    assert (right_factor.matmul_calls, right_factor.rmatmul_calls) == (0, 1)
    # Both directions feed the dense view, so the two agreement checks above are
    # evidence about the action and not about a single shared code path.
    assert torch.equal(mapped_left.dense(), torch.kron(complex_matrix, square))
    assert torch.equal(
        mapped_right.dense(),
        torch.kron(square, complex_matrix.transpose(0, 1).contiguous()),
    )


def test_equality_of_non_tensor_factors_is_by_value() -> None:
    matrix = torch.arange(9, dtype=torch.float64).reshape(_DIMENSION, _DIMENSION)

    assert SuperOperator.left_multiply(
        _MatrixFactor(matrix, tag="a")
    ) == SuperOperator.left_multiply(_MatrixFactor(matrix.clone(), tag="a"))
    assert SuperOperator.left_multiply(
        _MatrixFactor(matrix, tag="a")
    ) != SuperOperator.left_multiply(_MatrixFactor(matrix.clone(), tag="b"))
    assert SuperOperator.left_multiply(
        _MatrixFactor(matrix, tag="a")
    ) != SuperOperator.left_multiply(_MatrixFactor(matrix + 1.0, tag="a"))
    # A tensor factor and a protocol factor that carry the same matrix are still
    # distinct, because only two tensors take the tensor branch.
    assert SuperOperator.left_multiply(_MatrixFactor(matrix, tag="a")) != (
        SuperOperator.left_multiply(matrix.to(_COMPLEX128))
    )


class _MatrixWithoutDtype:
    """A matrix-like factor that omits the dtype the contract requires."""

    dimension = 3

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def dense(self) -> torch.Tensor:
        return torch.eye(3, dtype=_COMPLEX128)


def test_a_non_square_or_non_complex_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a square matrix"):
        SuperOperator.left_multiply(torch.zeros(2, 3, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must be a square matrix"):
        SuperOperator.left_multiply(torch.zeros(2, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must have a complex dtype"):
        SuperOperator.left_multiply(torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="must have a complex dtype"):
        SuperOperator.left_multiply(torch.eye(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="must support __matmul__"):
        SuperOperator.left_multiply(object())
    with pytest.raises(ValueError, match="must expose a torch dtype"):
        SuperOperator.left_multiply(_MatrixWithoutDtype())
    with pytest.raises(ValueError, match="positive integer dimension"):
        SuperOperator.left_multiply(
            type("_Bad", (_MatrixWithoutDtype,), {"dimension": 0})()
        )


def test_applying_a_state_of_the_wrong_shape_or_dtype_is_refused() -> None:
    applied = SuperOperator.left_multiply(_complex_factors()[0])

    with pytest.raises(ValueError, match="must match the superoperator dtype"):
        applied.apply(torch.zeros(_DIMENSION, _DIMENSION, dtype=_COMPLEX64))
    with pytest.raises(ValueError, match="must have shape"):
        applied.apply(torch.zeros(2, 2, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must have shape"):
        applied.apply(torch.zeros(_DIMENSION, dtype=_COMPLEX128))
    with pytest.raises(ValueError, match="must be a tensor"):
        applied.apply([1.0, 0.0])


def _general_map() -> SuperOperator:
    """Return a map with a two-sided, a left-only and a right-only term.

    The coefficients are complex and the factors are not normal, so conjugating
    the coefficient and conjugating a factor are both separately visible.
    """

    first, second, third, fourth = _complex_factors()
    mapped = SuperOperator.left_right_multiply(first, second) * complex(0.7, -1.3)
    mapped += SuperOperator.left_multiply(third) * complex(-0.25, 0.85)
    mapped += SuperOperator.right_multiply(fourth) * complex(0.45, 0.6)
    return mapped


def _probe_matrix(seed: int) -> torch.Tensor:
    """Return a unit-trace complex matrix, the state a pairing is evaluated on."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(_DIMENSION, _DIMENSION, dtype=_COMPLEX128, generator=generator)
    state = raw @ raw.conj().T
    return state / state.trace()


def _wrong_adjoint(mapped: SuperOperator, rule: str) -> SuperOperator:
    """Return a plausible-but-wrong adjoint, built from the same term list.

    Three rules are available, each keeping the coefficients so that the
    comparison is about the rule and not about a dropped scale: ``transpose``
    transposes each factor without conjugating it, ``swap`` moves each factor to
    the other side of the state, and ``conjugate`` leaves the coefficient alone.
    """

    wrong = SuperOperator()
    for value, left, right in mapped.terms:
        if rule == "conjugate":
            scale = value
        else:
            scale = value.conjugate()
        if rule == "swap":
            left, right = right, left
        if rule == "transpose":
            left = None if left is None else left.T
            right = None if right is None else right.T
        else:
            left = None if left is None else left.conj().T
            right = None if right is None else right.conj().T
        if left is None:
            wrong += SuperOperator.right_multiply(right) * scale
        elif right is None:
            wrong += SuperOperator.left_multiply(left) * scale
        else:
            wrong += SuperOperator.left_right_multiply(left, right) * scale
    return wrong


@pytest.mark.parametrize(
    ("rho_seed", "sigma_seed"), [(21, 22), (3, 4), (7, 8), (11, 12)]
)
def test_the_adjoint_satisfies_the_hilbert_schmidt_pairing(
    rho_seed: int, sigma_seed: int
) -> None:
    mapped = _general_map()
    adjoint = mapped.adjoint()
    rho, sigma = _probe_matrix(rho_seed), _probe_matrix(sigma_seed)
    inner = lambda left, right: (left.conj().T @ right).trace()  # noqa: E731

    forward = inner(sigma, mapped.apply(rho))
    backward = inner(adjoint.apply(sigma), rho)

    # The pairing is complex, so a floor on its modulus and a tolerance on the
    # difference are two separate statements. Measured over the four probe pairs
    # here, the moduli run from 0.53 to 4.52 and every difference is below
    # 1.0e-15, which is the round-off floor of the two evaluations.
    assert float(abs(forward)) > 0.5
    assert float(abs(forward - backward)) < 1e-14
    # Controls: the map itself does not pair with its own forward action, and
    # neither does any of the three wrong rules, so the identity above is not
    # satisfied by an arbitrary linear map. Measured residuals run from 4.9e-1
    # to 7.8e0 against the 1e-2 floor used here.
    assert float(abs(inner(mapped.apply(sigma), rho) - forward)) > 1e-2
    for rule in ("transpose", "swap", "conjugate"):
        wrong = _wrong_adjoint(mapped, rule)
        assert wrong != adjoint
        assert float(abs(inner(wrong.apply(sigma), rho) - forward)) > 1e-2


def test_the_adjoint_is_the_conjugate_transpose_of_the_dense_form() -> None:
    mapped = _general_map()
    adjoint = mapped.adjoint()

    # The oracle is the dense form built from the *action*, so the claim below
    # is a statement about the adjoint's own terms and not about a shared
    # Kronecker convention. The two routes agree at the round-off floor; the
    # measured worst entry is 1.99e-15.
    oracle = _matrix_from_action(mapped.terms, _DIMENSION, _COMPLEX128)
    dense_adjoint = adjoint.dense()

    assert float((dense_adjoint - oracle.conj().T).abs().max()) < 1e-14
    # Controls. A plain transpose is a different operator (measured 15.25), and
    # so are the map itself (11.37) and each wrong rule (7.72, 8.59, 14.30).
    assert float((dense_adjoint - oracle.T).abs().max()) > 1e-2
    assert float((dense_adjoint - mapped.dense()).abs().max()) > 1e-2
    for rule in ("transpose", "swap", "conjugate"):
        wrong = _wrong_adjoint(mapped, rule)
        assert float((dense_adjoint - wrong.dense()).abs().max()) > 1e-2


def test_the_adjoint_conjugates_the_coefficient_and_both_factors() -> None:
    mapped = _general_map()
    adjoint = mapped.adjoint()

    assert adjoint.dimension == mapped.dimension
    assert adjoint.dtype == mapped.dtype
    assert len(adjoint) == len(mapped)
    for (value, left, right), (adj_value, adj_left, adj_right) in zip(
        mapped.terms, adjoint.terms, strict=True
    ):
        assert adj_value == value.conjugate()
        # A one-sided term stays one-sided on the side it already occupied: the
        # adjoint of ``rho -> A rho`` is ``sigma -> A^dag sigma``, not
        # ``sigma -> sigma A^dag``.
        assert (left is None) == (adj_left is None)
        assert (right is None) == (adj_right is None)
        for original, derived in ((left, adj_left), (right, adj_right)):
            if original is not None:
                assert torch.equal(derived, original.conj().T)


def test_the_adjoint_is_an_involution_and_commutes_with_the_algebra() -> None:
    mapped = _general_map()
    other = SuperOperator.right_multiply(_complex_factors()[0])
    scalar = complex(-1.75, 0.5)

    # Every expectation below is exact rather than close: conjugation and scalar
    # multiplication of a complex number round in the same order on both sides.
    assert mapped.adjoint().adjoint() == mapped
    assert (mapped * scalar).adjoint() == mapped.adjoint() * scalar.conjugate()
    assert (mapped + other).adjoint() == mapped.adjoint() + other.adjoint()
    # An empty sum has no dimension to invent, so its adjoint is empty too.
    assert SuperOperator().adjoint() == SuperOperator()
    assert len(SuperOperator().adjoint()) == 0


class _MatrixWithoutAdjoint:
    """A factor that builds and applies a term but cannot be adjointed."""

    dtype = _COMPLEX128
    dimension = _DIMENSION

    def __init__(self, matrix: torch.Tensor) -> None:
        self._matrix = matrix.to(_COMPLEX128)

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor:
        return self._matrix @ state

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self._matrix

    def dense(self) -> torch.Tensor:
        return self._matrix


def test_a_factor_that_cannot_be_adjointed_is_refused_only_by_the_adjoint() -> None:
    matrix = torch.arange(9, dtype=torch.float64).reshape(_DIMENSION, _DIMENSION)
    applied = SuperOperator.left_multiply(_MatrixWithoutAdjoint(matrix))
    state = _density_matrix()

    # The refusal is the adjoint's own requirement: the factor satisfies every
    # other member, so construction and both views still work.
    assert applied.dimension == _DIMENSION
    assert torch.equal(applied.apply(state), matrix.to(_COMPLEX128) @ state)
    assert torch.equal(
        applied.dense(), torch.kron(matrix.to(_COMPLEX128), torch.eye(_DIMENSION))
    )
    with pytest.raises(ValueError, match="must support adjoint"):
        applied.adjoint()


class _AdjointCountingFactor(_MatrixWithoutAdjoint):
    """A non-tensor factor whose ``adjoint`` is counted and returns a new factor."""

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__(matrix)
        self.adjoint_calls = 0

    def adjoint(self) -> _AdjointCountingFactor:
        self.adjoint_calls += 1
        return _AdjointCountingFactor(self._matrix.conj().T)


def test_a_non_tensor_factor_is_adjointed_through_its_own_method() -> None:
    matrix = torch.arange(9, dtype=torch.float64).reshape(_DIMENSION, _DIMENSION)
    factor = _AdjointCountingFactor(matrix)
    applied = SuperOperator.left_multiply(factor)

    adjoint = applied.adjoint()

    assert factor.adjoint_calls == 1
    # The result is the factor's own adjoint, compared by value through the
    # dense view, so a dispatch that conjugated the matrix itself would pass
    # only by coincidence -- and here the matrix is real, so it would not.
    assert torch.equal(adjoint.dense(), applied.dense().conj().T)
    assert len(adjoint.terms) == 1
    assert adjoint.terms[0][1].adjoint_calls == 0


def test_the_dense_form_follows_the_factor_dtype() -> None:
    narrow = SuperOperator.left_multiply(torch.eye(_DIMENSION, dtype=_COMPLEX64))
    wide = SuperOperator.left_multiply(torch.eye(_DIMENSION, dtype=_COMPLEX128))

    matrix = narrow.dense()

    assert narrow.dtype == _COMPLEX64
    assert matrix.dtype == _COMPLEX64
    assert matrix.shape == (_DIMENSION**2, _DIMENSION**2)
    assert narrow.apply(torch.eye(_DIMENSION, dtype=_COMPLEX64)).dtype == _COMPLEX64
    assert torch.allclose(matrix, wide.dense().to(_COMPLEX64), atol=0.0)
    # complex64 is half the width, which is what makes the ceiling arithmetic
    # depend on the factor dtype rather than on a fixed constant.
    assert _COMPLEX64.itemsize == 8
    assert matrix.numel() * _COMPLEX64.itemsize == 648
    # An explicit ceiling strictly between the two widths separates them: the
    # 9 x 9 wide map is 81 entries * 16 = 1296 bytes and the narrow one is
    # 648, so a ceiling that ignored the dtype width would refuse both.
    assert narrow.dense(max_bytes=1000).numel() * _COMPLEX64.itemsize == 648
    with pytest.raises(ValueError, match="1296 bytes"):
        wide.dense(max_bytes=1000)


def test_a_factor_whose_dense_is_not_a_tensor_is_refused() -> None:
    class _BadDense:
        dtype = _COMPLEX128
        dimension = 3

        def __matmul__(self, state: torch.Tensor) -> torch.Tensor:
            return state

        def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor:
            return state

        def dense(self) -> object:
            return [[1.0]]

    applied = SuperOperator.left_multiply(_BadDense())

    # The factor passes the protocol check, so the refusal comes from the view.
    assert applied.dimension == 3
    with pytest.raises(ValueError, match="must return a tensor from dense"):
        applied.dense()


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


def _normalized_pauli_basis(
    n_qubits: int, dtype: torch.dtype = _COMPLEX128
) -> tuple[torch.Tensor, ...]:
    """Return ``P_i / sqrt(d)`` for one or two wires, built without the network.

    The four matrices are written out as literals rather than read from
    ``flagquantum.simulation.pauli``, so a convention error in the Pauli-product
    helper cannot be inherited by the basis this module transforms with. The word
    order is the identity-first order ``flagquantum.simulation.pauli.pauli_words``
    fixes and ``flagquantum.algorithms.pec`` indexes its weights by.
    """

    identity = torch.eye(2, dtype=dtype)
    exchange = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    # ``Y`` is ``i`` times the real antisymmetric matrix. The factor matters even
    # though a per-element phase leaves the diagonal of `P^dag D P` untouched: it
    # is what makes this the Pauli basis rather than a basis that merely resembles
    # it, and only the literal form can catch a phase error on the production
    # builder this helper is the independent counterweight to.
    imaginary = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype) * 1j
    phase = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    single = (identity, exchange, imaginary, phase)
    if n_qubits == 1:
        return tuple(matrix / (2.0**0.5) for matrix in single)
    if n_qubits != 2:
        raise ValueError(f"this helper builds one or two wires, not {n_qubits}")
    return tuple(torch.kron(left, right) / 2.0 for left in single for right in single)


def _basis_action_oracle(
    mapped: SuperOperator, basis: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Build ``trace(B_i^dag E(B_j))`` from the map's own ``apply``."""

    size = len(basis)
    matrix = torch.zeros(size, size, dtype=basis[0].dtype)
    for row in range(size):
        for column in range(size):
            matrix[row, column] = torch.trace(
                basis[row].mH @ mapped.apply(basis[column])
            )
    return matrix


def _from_kraus_by_hand(operators: tuple[torch.Tensor, ...]) -> SuperOperator:
    """Accumulate ``sum_k K_k . K_k^dag`` through the public constructors."""

    total = SuperOperator()
    for operator in operators:
        total += SuperOperator.left_right_multiply(operator, operator.conj().T)
    return total


def test_from_kraus_is_the_two_sided_sum_of_its_operators() -> None:
    operators = _complex_factors(_DIMENSION)
    mapped = SuperOperator.from_kraus(operators)
    expected = _from_kraus_by_hand(operators)
    state = _density_matrix()

    assert mapped == expected
    assert len(mapped) == len(operators)
    assert torch.equal(mapped.apply(state), expected.apply(state))
    assert torch.equal(mapped.dense(), expected.dense())
    # Each operator is used beside its own adjoint, and the right factor is the
    # adjoint rather than the operator, so a non-normal pair is distinguishable.
    for index, operator in enumerate(operators):
        coefficient, left, right = mapped.terms[index]
        assert coefficient == 1.0 + 0.0j
        assert left is operator
        assert torch.equal(right, operator.conj().T)


def test_from_kraus_refuses_an_empty_list_by_name() -> None:
    with pytest.raises(ValueError) as refused:
        SuperOperator.from_kraus(())
    message = str(refused.value)
    assert "at least one operator" in message
    assert "no dimension to act on" in message


def test_from_kraus_refuses_a_factor_that_cannot_be_adjointed() -> None:
    matrix = torch.eye(_DIMENSION, dtype=torch.float64)
    with pytest.raises(ValueError, match="must support adjoint"):
        SuperOperator.from_kraus((_MatrixWithoutAdjoint(matrix),))


def test_from_kraus_refuses_operators_that_disagree_on_dimension() -> None:
    with pytest.raises(ValueError, match="share one dimension"):
        SuperOperator.from_kraus(
            (torch.eye(2, dtype=_COMPLEX128), torch.eye(4, dtype=_COMPLEX128))
        )


def _unit_matrices(dimension: int) -> tuple[torch.Tensor, ...]:
    """Return the ``d**2`` matrix units, an orthonormal basis of matrices."""

    matrices = []
    for row in range(dimension):
        for column in range(dimension):
            matrix = torch.zeros(dimension, dimension, dtype=_COMPLEX128)
            matrix[row, column] = 1.0 + 0.0j
            matrices.append(matrix)
    return tuple(matrices)


def test_matrix_in_basis_of_the_identity_map_is_the_identity() -> None:
    for n_qubits in (1, 2):
        dimension = 2**n_qubits
        identity_map = SuperOperator.from_kraus(
            (torch.eye(dimension, dtype=_COMPLEX128),)
        )
        expected = torch.eye(dimension * dimension, dtype=_COMPLEX128)

        transfer = identity_map.matrix_in_basis(_normalized_pauli_basis(n_qubits))
        assert transfer.shape == (dimension * dimension, dimension * dimension)
        assert torch.allclose(transfer, expected, atol=1e-14)

        # The matrix units are orthonormal too, and in them the identity map's
        # matrix is the identity for the same reason rather than by coincidence.
        assert torch.allclose(
            identity_map.matrix_in_basis(_unit_matrices(dimension)),
            expected,
            atol=1e-14,
        )


def test_matrix_in_basis_agrees_with_the_maps_own_action() -> None:
    for n_qubits in (1, 2):
        dimension = 2**n_qubits
        operators = (
            _complex_factors(dimension)[0],
            _complex_factors(dimension)[1],
        )
        mapped = SuperOperator.from_kraus(operators)
        basis = _normalized_pauli_basis(n_qubits)
        assert torch.allclose(
            mapped.matrix_in_basis(basis),
            _basis_action_oracle(mapped, basis),
            atol=1e-13,
        )


def test_matrix_in_basis_reads_the_pauli_transfer_matrix_of_a_channel() -> None:
    identity = torch.eye(2, dtype=_COMPLEX128)
    exchange = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=_COMPLEX128)
    phase = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=_COMPLEX128)
    basis = _normalized_pauli_basis(1)

    # A bit flip at probability p sends X to (1 - 2p) X and Y to (1 - 2p) Y while
    # leaving I and Z alone, so its transfer matrix is diagonal.
    probability = 0.25
    flip = SuperOperator.from_kraus(
        ((1.0 - probability) ** 0.5 * identity, probability**0.5 * exchange)
    )
    expected = torch.diag(
        torch.tensor([1.0, 1.0, 1.0 - 2.0 * probability, 1.0 - 2.0 * probability])
    ).to(_COMPLEX128)
    assert torch.allclose(flip.matrix_in_basis(basis), expected, atol=1e-14)

    # A phase flip is the same statement on the other two axes.
    damped = SuperOperator.from_kraus(
        ((1.0 - probability) ** 0.5 * identity, probability**0.5 * phase)
    )
    expected_phase = torch.diag(
        torch.tensor([1.0, 1.0 - 2.0 * probability, 1.0 - 2.0 * probability, 1.0])
    ).to(_COMPLEX128)
    assert torch.allclose(damped.matrix_in_basis(basis), expected_phase, atol=1e-14)

    # Amplitude damping is not a Pauli channel: its only off-diagonal entry is the
    # transition the map performs, and that entry is exactly gamma.
    gamma = 0.3
    lowering = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=_COMPLEX128)
    damping = SuperOperator.from_kraus(
        (
            torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma) ** 0.5]], dtype=_COMPLEX128),
            (gamma**0.5) * lowering,
        )
    )
    transfer = damping.matrix_in_basis(basis)
    expected_damping = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, (1.0 - gamma) ** 0.5, 0.0, 0.0],
            [0.0, 0.0, (1.0 - gamma) ** 0.5, 0.0],
            [gamma, 0.0, 0.0, 1.0 - gamma],
        ],
        dtype=_COMPLEX128,
    )
    assert torch.allclose(transfer, expected_damping, atol=1e-15)


def test_matrix_in_basis_admits_a_single_precision_basis() -> None:
    # The tolerance is eight epsilons of the map's own dtype, so the same
    # normalized Pauli basis that is orthonormal to 2.2e-16 in complex128 is
    # admitted in complex64, where its residual is 6.0e-08. A single absolute
    # threshold would have refused one of the two.
    single = _normalized_pauli_basis(1, torch.complex64)
    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=torch.complex64),))
    transfer = mapped.matrix_in_basis(single)
    assert transfer.dtype == torch.complex64
    assert torch.allclose(transfer, torch.eye(4, dtype=torch.complex64), atol=1e-6)


def test_matrix_in_basis_refuses_a_basis_that_is_not_the_right_length() -> None:
    basis = _normalized_pauli_basis(1)
    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=_COMPLEX128),))

    with pytest.raises(ValueError) as short:
        mapped.matrix_in_basis(basis[:3])
    assert "basis of 4 matrices" in str(short.value)
    assert "3 were given" in str(short.value)

    with pytest.raises(ValueError) as long:
        mapped.matrix_in_basis((*basis, basis[0]))
    assert "5 were given" in str(long.value)


def test_matrix_in_basis_refuses_a_basis_that_is_not_orthonormal() -> None:
    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=_COMPLEX128),))
    units = _normalized_pauli_basis(1)

    # The unnormalized Pauli words span the same space and are the natural wrong
    # basis to reach for: their Gram matrix is 2 * the identity, exactly 1.0 away.
    unnormalized = tuple(matrix * (2.0**0.5) for matrix in units)
    with pytest.raises(ValueError) as refused:
        mapped.matrix_in_basis(unnormalized)
    message = str(refused.value)
    assert "not orthonormal" in message
    assert "1.000e+00 away from the identity" in message
    assert "1.776e-15 tolerance" in message
    assert "P^dag D P rather than P^-1 D P" in message

    # A basis that is almost orthonormal is still refused, because the residual is
    # what the returned matrix is wrong by and the margin here is one part in 1e-9.
    almost = list(units)
    almost[1] = almost[1] * (1.0 + 1e-6)
    with pytest.raises(ValueError, match="2.000e-06 away from the identity"):
        mapped.matrix_in_basis(almost)


def test_matrix_in_basis_refuses_an_entry_that_is_not_a_basis_matrix() -> None:
    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=_COMPLEX128),))
    units = _normalized_pauli_basis(1)

    with pytest.raises(ValueError, match="must be a tensor"):
        mapped.matrix_in_basis((*units[:3], None))
    with pytest.raises(ValueError, match="must be a 2 x 2 matrix"):
        mapped.matrix_in_basis((*units[:3], torch.eye(3, dtype=_COMPLEX128)))
    with pytest.raises(ValueError, match="must share the superoperator dtype"):
        mapped.matrix_in_basis((*units[:3], units[3].to(torch.complex64)))
    with pytest.raises(ValueError, match="needs at least one element"):
        mapped.matrix_in_basis(())


def test_matrix_in_basis_refuses_a_basis_on_another_device() -> None:
    # The dense form is a CPU reference, so a basis that is not on the CPU is
    # refused by name rather than reaching a Kronecker product whose operands
    # disagree. ``meta`` supplies the second device without an accelerator.
    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=_COMPLEX128),))
    elsewhere = tuple(matrix.to(device="meta") for matrix in _normalized_pauli_basis(1))

    with pytest.raises(ValueError) as refused:
        mapped.matrix_in_basis(elsewhere)
    message = str(refused.value)
    assert "must live on the CPU device" in message
    assert "this one is on meta" in message


def test_the_published_pauli_basis_composes_with_the_basis_transform() -> None:
    """The seam between the two modules: the basis Core reads is Simulation's.

    ``_normalized_pauli_basis`` above is this file's own oracle, built from
    literals so it cannot inherit a convention error from the production path.
    This test is the other direction: it feeds the basis
    :func:`flagquantum.simulation.pauli.normalized_pauli_basis` publishes into
    :meth:`SuperOperator.matrix_in_basis` and checks the same transfer matrix
    comes back, so the two modules agree on the word order and the scale rather
    than only each agreeing with this file.
    """

    from flagquantum.simulation.pauli import normalized_pauli_basis, pauli_words

    assert pauli_words(1) == ("I", "X", "Y", "Z")
    published = normalized_pauli_basis(1, dtype=_COMPLEX128, device="cpu")
    mine = _normalized_pauli_basis(1)
    assert len(published) == len(mine)
    for expected, actual in zip(mine, published, strict=True):
        assert torch.allclose(expected, actual, atol=0.0)

    probability = 0.25
    flip = SuperOperator.from_kraus(
        (
            (1.0 - probability) ** 0.5 * torch.eye(2, dtype=_COMPLEX128),
            probability**0.5
            * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=_COMPLEX128),
        )
    )
    expected = torch.diag(
        torch.tensor([1.0, 1.0, 1.0 - 2.0 * probability, 1.0 - 2.0 * probability])
    ).to(_COMPLEX128)
    assert torch.allclose(flip.matrix_in_basis(published), expected, atol=1e-14)


def test_matrix_in_basis_refuses_a_basis_spanning_two_devices() -> None:
    """A mixed basis is refused by name, not by ``torch.stack``'s own error.

    ``torch.stack`` refuses tensors that disagree on device with a bare
    ``RuntimeError``, so the named refusal has to run before the stack rather than
    after it. The assertion is on the message, because the ordering is the whole
    claim: if the stack ran first the caller would see the tensor library's
    wording and this message would never be produced.
    """

    mapped = SuperOperator.from_kraus((torch.eye(2, dtype=_COMPLEX128),))
    basis = _normalized_pauli_basis(1)
    mixed = (basis[0], basis[1], basis[2], basis[3].to(device="meta"))

    with pytest.raises(ValueError) as refused:
        mapped.matrix_in_basis(mixed)
    message = str(refused.value)
    assert "must share one device" in message
    assert "cpu" in message
    assert "meta" in message


def _choi_action_oracle(mapped: SuperOperator, dimension: int) -> torch.Tensor:
    """Build ``sum_ab E_ab (x) E(E_ab)`` from the map's own ``apply``.

    This is the Choi matrix by its definition rather than by a reshape, so it
    cannot inherit an index-order error from the implementation. ``apply`` is
    called with the whole stack of matrix units at once, which is the same call
    the network makes and therefore a different path through the factor algebra
    than ``dense`` is.
    """

    units = torch.stack(_unit_matrices(dimension))
    images = mapped.apply(units)
    return torch.stack(
        [
            torch.kron(units[index], images[index])
            for index in range(dimension * dimension)
        ]
    ).sum(0)


def _transpose_map(
    dimension: int, scale: float = 1.0, dtype: torch.dtype = _COMPLEX128
) -> SuperOperator:
    """Return ``rho -> scale * rho^T``, the standard positive-but-not-CP map."""

    total = SuperOperator()
    for row in range(dimension):
        for column in range(dimension):
            matrix = torch.zeros(dimension, dimension, dtype=dtype)
            matrix[row, column] = 1.0 + 0.0j
            total += SuperOperator.left_right_multiply(matrix, matrix) * scale
    return total


def _amplitude_damping(dimension: int, gamma: float = 0.4) -> SuperOperator:
    """The ``d``-level amplitude damping channel, trace preserving by construction.

    The lowering part acts on every level above the ground state, so ``K_1^dag K_1``
    is ``gamma`` times the projector onto those levels and the two Kraus operators
    together sum to the identity. A single ``|0><1|`` term would leave every higher
    level untouched, and the map would then not be trace preserving at all -- which
    is a property this file asserts, so the fixture has to have it.
    """

    diagonal = torch.diag(
        torch.tensor(
            [1.0] + [(1.0 - gamma) ** 0.5] * (dimension - 1), dtype=torch.complex128
        )
    )
    lowering = torch.zeros(dimension, dimension, dtype=torch.complex128)
    for level in range(1, dimension):
        lowering[level - 1, level] = 1.0
    return SuperOperator.from_kraus((diagonal, gamma**0.5 * lowering))


def _rotated_dephasing(
    dimension: int, scale: float, dtype: torch.dtype
) -> SuperOperator:
    """A trace-preserving, rank-deficient channel whose Choi entries are inexact.

    The plain dephasing channel's Choi matrix is diagonal, so scaling it produces
    exactly representable entries and no rounding to measure. Conjugating it by a
    unitary that is not a permutation keeps the rank deficiency while making every
    entry a sum of products, which is what makes the rounding visible.
    """

    generator = torch.Generator().manual_seed(20261011)
    real = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    imaginary = torch.randn(
        dimension, dimension, dtype=torch.float64, generator=generator
    )
    q, r = torch.linalg.qr((real + 1j * imaginary).to(dtype))
    diagonal = torch.diagonal(r)
    unitary = q * (diagonal / diagonal.abs()).conj()

    operators = []
    for index in range(dimension):
        projector = torch.zeros(dimension, dimension, dtype=dtype)
        projector[index, index] = 1.0
        operators.append(scale * unitary @ projector @ unitary.conj().transpose(0, 1))
    return SuperOperator.from_kraus(tuple(operators))


def _smallest_choi_eigenvalue(mapped: SuperOperator) -> tuple[float, float]:
    """Return the smallest Choi eigenvalue and the Choi matrix's spectral scale."""

    choi = mapped.choi()
    eigenvalues = torch.linalg.eigvalsh((choi + choi.conj().transpose(0, 1)) / 2)
    return float(eigenvalues.min()), float(eigenvalues.abs().max())


def test_choi_is_the_action_built_matrix_in_the_declared_index_order() -> None:
    """The convention is pinned at a dimension where the candidates disagree.

    Every index order that keeps the four indices in pairs gives a Hermitian
    matrix with the map's trace along the diagonal, so a trace check cannot tell
    them apart and neither can one wire: at ``d = 2`` two of the three wrong orders
    coincide with the right one. Three wires is the smallest dimension where all
    three wrong orders are separated, so that is where the assertion is made.
    """

    probability = 0.3
    for dimension in (2, 3, 4):
        identity = torch.eye(dimension, dtype=_COMPLEX128)
        shift = torch.roll(identity, shifts=1, dims=0)
        mapped = SuperOperator.from_kraus(
            (
                (1.0 - probability) ** 0.5 * identity,
                (probability / (dimension * dimension - 1)) ** 0.5 * shift,
            )
        )
        oracle = _choi_action_oracle(mapped, dimension)
        assert torch.allclose(mapped.choi(), oracle, atol=1e-14)

        if dimension < 3:
            continue
        shape = (dimension, dimension, dimension, dimension)
        for order in ((0, 2, 1, 3), (1, 0, 3, 2), (0, 2, 3, 1)):
            reshuffled = (
                mapped.dense()
                .reshape(shape)
                .permute(*order)
                .reshape(dimension * dimension, dimension * dimension)
            )
            assert float((reshuffled - oracle).abs().max()) > 1e-3


def test_choi_of_a_trace_preserving_map_has_trace_d() -> None:
    """A property of the map, so it holds for a map that is not even positive."""

    for dimension in (2, 3):
        identity = torch.eye(dimension, dtype=_COMPLEX128)
        channels = {
            "identity": SuperOperator.from_kraus((identity,)),
            "dephasing": SuperOperator.from_kraus(
                tuple(
                    torch.diag(
                        torch.tensor(
                            [
                                1.0 if index == other else 0.0
                                for other in range(dimension)
                            ],
                            dtype=_COMPLEX128,
                        )
                    )
                    for index in range(dimension)
                )
            ),
            "amplitude damping": _amplitude_damping(dimension),
            "transpose": _transpose_map(dimension),
        }
        for name, mapped in channels.items():
            choi = mapped.choi()
            assert choi.shape == (dimension * dimension, dimension * dimension), name
            # The empirical entry count is exact, not approximate: the trace is a
            # sum of the diagonal, and every channel here has exactly representable
            # diagonal entries.
            assert choi.trace() == complex(dimension), name
            assert torch.allclose(
                choi.trace().reshape(1),
                mapped.apply(identity).trace().reshape(1),
                atol=1e-14,
            ), name


def test_partial_trace_reads_the_maps_normalization_off_the_choi_matrix() -> None:
    """The input trace is ``E(I)`` and the output trace is the identity for a TP map.

    Both are properties of the same matrix rather than of two matrices, so they
    also say which of the two traces is which -- a swap would put the identity on
    the wrong axis and be invisible on a unital map.
    """

    for dimension in (2, 3):
        identity = torch.eye(dimension, dtype=_COMPLEX128)
        unital = SuperOperator.from_kraus((identity,))
        damping = _amplitude_damping(dimension)
        for mapped in (unital, damping, _transpose_map(dimension)):
            first = mapped.partial_trace("input")
            second = mapped.partial_trace("output")
            assert torch.allclose(first, mapped.apply(identity), atol=1e-14)
            # A trace-preserving map has the identity there, and both maps here
            # are trace preserving, so this is not a coincidence of the unital one.
            assert torch.allclose(second, identity, atol=1e-14)

        # Amplitude damping is the counterexample that separates the two partial
        # traces: its first one is ``diag(1 + gamma, 1 - gamma, ...)``, which is
        # ``E(I)`` and is not the identity. If the two were swapped in the
        # implementation, this is the assertion that would notice.
        first = damping.partial_trace("input")
        # ``K_0 I K_0^dag`` is ``diag(1, 1 - gamma, ..., 1 - gamma)`` and
        # ``K_1 I K_1^dag`` is ``gamma`` on the levels the lowering leaves behind,
        # so the first partial trace is ``diag(1 + gamma, 1, ..., 1, 1 - gamma)``.
        expected = torch.diag(
            torch.tensor(
                [1.0 + 0.4] + [1.0] * (dimension - 2) + [1.0 - 0.4],
                dtype=_COMPLEX128,
            )
        )
        assert torch.allclose(first, expected, atol=1e-15)
        assert float((first - identity).abs().max()) == pytest.approx(0.4)


def test_partial_trace_matches_the_action_it_summarizes() -> None:
    """Both registers against an oracle built from the map's own ``apply``.

    The oracle never reshapes ``choi()``: ``E(I)`` is one call to ``apply``, and
    ``trace(E(E_ab))`` is one call per matrix unit. Reading the same two matrices
    off the Choi index order is therefore a claim about that index order rather
    than a second transcription of it. The fixtures separate the two registers: the
    identity channel is both unital and trace preserving, while a one-sided action
    is neither, so a swap of the two registers or a transposed pair of indices
    would be visible here rather than hidden by a symmetric map.
    """

    for dimension in (2, 3, 4):
        identity = torch.eye(dimension, dtype=_COMPLEX128)
        units = _unit_matrices(dimension)
        generator = torch.Generator().manual_seed(11)
        random = torch.randn(
            dimension, dimension, dtype=torch.complex128, generator=generator
        )
        maps = {
            "identity": SuperOperator.from_kraus((identity,)),
            "amplitude damping": _amplitude_damping(dimension),
            "transpose": _transpose_map(dimension),
            # A one-sided action is a generator rather than a channel, and it is
            # the case where neither register's trace is the identity.
            "generator": SuperOperator.left_multiply(random),
        }
        for name, mapped in maps.items():
            expected_input = mapped.apply(identity)
            # ``_unit_matrices`` walks the units row-major, so reshaping the traces
            # by dimension indexes them as ``[a, b]``, which is ``E(E_ab)``.
            expected_output = torch.stack(
                [mapped.apply(unit).trace().reshape(()) for unit in units]
            ).reshape(dimension, dimension)
            assert torch.allclose(
                mapped.partial_trace("input"), expected_input, atol=1e-14
            ), name
            assert torch.allclose(
                mapped.partial_trace("output"), expected_output, atol=1e-14
            ), name
        # The generator is the fixture where both of the above are non-trivially
        # different from the identity, so it is the one that pins the reading.
        assert not torch.allclose(
            maps["generator"].partial_trace("input"), identity, atol=1e-6
        )


def test_partial_trace_keeps_the_maps_dtype_and_refuses_a_third_register() -> None:
    """The accessor adds no dtype, no device, and no register the map does not have."""

    for dtype in (_COMPLEX128, _COMPLEX64):
        identity = torch.eye(2, dtype=dtype)
        mapped = SuperOperator.from_kraus((identity,))
        for register in ("input", "output"):
            trace = mapped.partial_trace(register)
            assert trace.shape == (2, 2), register
            assert trace.dtype == dtype, register
            assert trace.device.type == "cpu", register

    damping = _amplitude_damping(3)
    for refused in ("Input", "inputs", "", "both", "output ", "in"):
        with pytest.raises(ValueError) as error:
            damping.partial_trace(refused)
        message = str(error.value)
        assert "'input'" in message and "'output'" in message
        # The refusal names what the operation is not, because a caller reaching
        # for this spelling usually wants a subsystem rather than a register.
        assert "not this operation" in message

    # The byte ceiling is ``choi``'s, because the trace is read off that matrix.
    four = SuperOperator.left_multiply(torch.eye(4, dtype=_COMPLEX128))
    assert four.partial_trace("input", max_bytes=4096).shape == (4, 4)
    with pytest.raises(ValueError) as error:
        four.partial_trace("output", max_bytes=4095)
    assert "4096 bytes" in str(error.value)


def test_partial_trace_is_differentiable_in_the_maps_factors() -> None:
    """Each register's entries carry a gradient a training loop can reach.

    The assertion is against a finite difference of the same entry rather than
    against hand algebra, and the entry is the ``[0, 0]`` one on a map whose
    off-diagonal factor entry depends on the parameter. For this map
    ``K K^dag`` and ``K^dag K`` agree in the sum of their real parts and differ
    entry by entry, so the two registers are checked on different numbers and one
    of them cannot be passing on the other's behalf.
    """

    def damping_strength(probability: float) -> SuperOperator:
        factor = torch.zeros(2, 2, dtype=_COMPLEX128)
        factor[0, 0] = (1.0 - probability) ** 0.5
        factor[0, 1] = 0.5 * probability
        factor[1, 1] = (1.0 - probability) ** 0.5
        return SuperOperator.from_kraus((factor,))

    def entry(register: str, probability: float) -> float:
        return float(damping_strength(probability).partial_trace(register).real[0, 0])

    step = 1e-6
    gradients = {}
    for register in ("input", "output"):
        parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        factor = torch.zeros(2, 2, dtype=_COMPLEX128)
        factor[0, 0] = torch.sqrt(1.0 - parameter)
        factor[0, 1] = 0.5 * parameter
        factor[1, 1] = torch.sqrt(1.0 - parameter)
        SuperOperator.from_kraus((factor,)).partial_trace(register).real[
            0, 0
        ].backward()
        assert parameter.grad is not None
        measured = float(parameter.grad)

        finite_difference = (
            entry(register, 0.3 + step) - entry(register, 0.3 - step)
        ) / (2 * step)
        assert measured == pytest.approx(finite_difference, abs=1e-9)
        gradients[register] = measured

    # The fixture separates the registers, so the two gradients above are two
    # measurements of two different quantities rather than one number twice.
    assert gradients["input"] == pytest.approx(-0.85, abs=1e-9)
    assert gradients["output"] == pytest.approx(-1.0, abs=1e-9)


def test_choi_is_refused_above_its_byte_ceiling() -> None:
    """The ceiling is the dense form's, because the matrix is the dense form."""

    four = SuperOperator.left_multiply(torch.eye(4, dtype=_COMPLEX128))
    assert four.choi(max_bytes=4096).shape == (16, 16)
    with pytest.raises(ValueError) as refused:
        four.choi(max_bytes=4095)
    message = str(refused.value)
    assert "16 x 16" in message
    assert "4096 bytes" in message


def test_is_completely_positive_accepts_every_kraus_channel() -> None:
    """Kraus lists denote completely positive maps by construction."""

    for dtype in (_COMPLEX128, _COMPLEX64):
        for dimension in (2, 3):
            identity = torch.eye(dimension, dtype=dtype)
            shift = torch.roll(identity, shifts=1, dims=0)
            phase = torch.diag(
                torch.tensor(
                    [(-1.0) ** index for index in range(dimension)], dtype=dtype
                )
            )
            probability = 0.3
            cases = {
                "identity": SuperOperator.from_kraus((identity,)),
                # Rank deficient: its Choi matrix has exact zeros, which is the
                # case a tolerance has to exist for in the first place.
                "dephasing": SuperOperator.from_kraus(
                    tuple(
                        torch.diag(
                            torch.tensor(
                                [
                                    1.0 if index == other else 0.0
                                    for other in range(dimension)
                                ],
                                dtype=dtype,
                            )
                        )
                        for index in range(dimension)
                    )
                ),
                "depolarizing": SuperOperator.from_kraus(
                    (
                        (1.0 - probability) ** 0.5 * identity,
                        (probability / (dimension * dimension - 1)) ** 0.5 * shift,
                        (probability / (dimension * dimension - 1)) ** 0.5 * phase,
                    )
                ),
                # A single Kraus operator, hence not trace preserving when it is
                # not an isometry, and still completely positive: the certificate
                # is about positivity and not about normalization. A one-sided
                # multiplication would be the wrong example here, because it does
                # not preserve Hermiticity at all.
                "single scaled operator": SuperOperator.from_kraus((0.5 * shift,)),
            }
            for name, mapped in cases.items():
                assert mapped.is_completely_positive(), f"{name} in {dtype}"
                assert _smallest_choi_eigenvalue(mapped)[0] > -1e-6, name


def test_is_completely_positive_refuses_the_transpose_map() -> None:
    """The positive-but-not-completely-positive map, refused on positivity alone.

    The transpose map is trace preserving and its Choi matrix is Hermitian, so
    neither the trace nor the Hermiticity check can be what refuses it -- and the
    eigenvalue that does is exactly minus one, not a rounding residue.
    """

    for dtype in (_COMPLEX128, _COMPLEX64):
        for dimension in (2, 3):
            mapped = _transpose_map(dimension, dtype=dtype)
            identity = torch.eye(dimension, dtype=dtype)
            # It really is the transpose, and it really is a legitimate map.
            for index in range(dimension):
                for other in range(dimension):
                    unit = torch.zeros(dimension, dimension, dtype=dtype)
                    unit[index, other] = 1.0
                    assert torch.allclose(
                        mapped.apply(unit), unit.transpose(0, 1), atol=1e-15
                    )
            assert torch.allclose(mapped.apply(identity), identity, atol=1e-15)
            assert mapped.choi().trace() == complex(dimension)

            smallest, scale = _smallest_choi_eigenvalue(mapped)
            assert smallest == pytest.approx(-1.0, abs=1e-6)
            assert scale == pytest.approx(1.0, abs=1e-6)
            assert mapped.is_completely_positive() is False


def test_is_completely_positive_refuses_a_map_that_does_not_preserve_hermiticity() -> (
    None
):
    """``False`` would be a false statement about a map that is not a candidate."""

    for dtype in (_COMPLEX128, _COMPLEX64):
        generator = SuperOperator.left_multiply(
            1j * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        )
        with pytest.raises(ValueError) as refused:
            generator.is_completely_positive()
        message = str(refused.value)
        assert "does not preserve Hermiticity" in message
        assert "Choi matrix is" in message
        assert "has no positivity to report" in message

        # And the same matrix that is refused really is not Hermitian, so the
        # refusal is not a tolerance that is simply too tight.
        choi = generator.choi()
        assert float((choi - choi.conj().transpose(0, 1)).abs().max()) > 1e-3

        # A Hermiticity-preserving map of the same shape is answered rather than
        # refused, so the branch is about this map and not about the shape.
        assert SuperOperator.from_kraus(
            (torch.eye(2, dtype=dtype),)
        ).is_completely_positive()


def test_the_complete_positivity_tolerance_is_relative_to_the_choi_scale() -> None:
    """The tolerance's scale factor is load bearing, measured where it decides.

    A rank-deficient channel scaled by ``c`` has a smallest Choi eigenvalue of
    exactly zero in exact arithmetic, so every deviation here is rounding and the
    ratio of that deviation to the matrix's own spectral scale is a constant of the
    channel. The absolute deviation, in epsilons, is not: it grows with ``c**2``,
    and by the top of this sweep it is four orders past what an absolute threshold
    of eight epsilons would permit. A tolerance stated in absolute epsilons would
    refuse a map that is exactly on the positivity boundary.
    """

    dimension = 4
    identity = torch.eye(dimension, dtype=_COMPLEX128)
    for dtype in (_COMPLEX128, _COMPLEX64):
        epsilon = float(torch.finfo(dtype).eps)
        ratios = []
        absolute = []
        for scale in (1.0, 4.0, 16.0, 256.0, 1024.0):
            channel = _rotated_dephasing(dimension, scale, dtype)
            smallest, spectral = _smallest_choi_eigenvalue(channel)
            assert channel.is_completely_positive()
            # The map is the same trace-preserving channel multiplied by
            # ``scale**2``, which is exactly why the ratio below is constant: the
            # deviation and the scale it is measured against are rounded from the
            # same entries.
            rescaled = channel.apply(identity.to(dtype))
            deviation = float((rescaled - scale**2 * identity.to(dtype)).abs().max())
            assert deviation < 1e-6 * scale**2
            ratios.append(-smallest / (epsilon * spectral))
            absolute.append(-smallest / epsilon)
        # The relative deviation is a constant of the channel, so the certificate
        # reads the same number at every scale in the sweep. Nothing but the Choi
        # scale turns it into a constant; in absolute units the same sweep spans
        # five orders of magnitude.
        assert max(ratios) - min(ratios) < 1e-9
        assert max(abs(ratio) for ratio in ratios) < 2.0
        assert max(absolute) > 8.0

        # The same scaling applied to a genuinely non-positive map stays refused,
        # so the relative tolerance did not simply widen the accepted set.
        scaled = _transpose_map(2, scale=1024.0, dtype=dtype)
        assert scaled.is_completely_positive() is False
