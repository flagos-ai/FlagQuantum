"""Source assertion semantics preserve phase and norm distinctions."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.matrix_assertions import (
    assert_matrices_almost_equal,
    assert_unitaries_equivalent_upto_global_phase,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 2), (0, 0)])
@pytest.mark.parametrize("atol", [-1.0, 0.0, 1e-5, 0.125])
def test_phase_sensitive_norm_matches_source(
    shape: tuple[int, ...], atol: float
) -> None:
    from qualtran.linalg.testing import assert_matrices_almost_equal as source

    left = np.zeros(shape, dtype=np.complex128)
    right = left.copy()
    if right.size:
        right.flat[0] = 0.125j
    before = right.copy()
    if np.linalg.norm(left - right) <= atol:
        source(left, right, atol=atol)
        assert_matrices_almost_equal(left, right, atol=atol)
    else:
        with pytest.raises(AssertionError) as original:
            source(left, right, atol=atol)
        with pytest.raises(AssertionError) as native:
            assert_matrices_almost_equal(left, right, atol=atol)
        assert str(native.value) == str(original.value)
    np.testing.assert_array_equal(right, before)


@pytest.mark.parametrize("phase", [0.0, 0.31, np.pi])
@pytest.mark.parametrize("relative", [False, True])
def test_unitary_assertion_matches_source(phase: float, relative: bool) -> None:
    from qualtran.linalg.testing import (
        assert_unitaries_equivalent_upto_global_phase as source,
    )

    rng = np.random.RandomState(17)
    left = np.linalg.qr(rng.randn(3, 3) + 1j * rng.randn(3, 3))[0]
    right = np.exp(1j * phase) * left
    if relative:
        right[:, 0] *= -1
        with pytest.raises(AssertionError) as original:
            source(left, right)
        with pytest.raises(AssertionError) as native:
            assert_unitaries_equivalent_upto_global_phase(left, right)
        assert str(native.value) == str(original.value)
    else:
        source(left, right)
        assert_unitaries_equivalent_upto_global_phase(left, right)


@pytest.mark.parametrize("shapes", [((2, 2), (3, 3)), ((2, 3), (2, 3)), ((2,), (2,))])
def test_shape_errors_and_order_match_source(
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    from qualtran.linalg.testing import assert_matrices_almost_equal as source_matrix
    from qualtran.linalg.testing import (
        assert_unitaries_equivalent_upto_global_phase as source_unitary,
    )

    left, right = (np.zeros(shape) for shape in shapes)
    if left.shape != right.shape:
        for check in (source_matrix, assert_matrices_almost_equal):
            with pytest.raises(AssertionError):
                check(left, right)
    else:
        source_matrix(left, right)
        assert_matrices_almost_equal(left, right)
    with pytest.raises(ValueError) as original:
        source_unitary(left, right)
    with pytest.raises(ValueError) as native:
        assert_unitaries_equivalent_upto_global_phase(left, right)
    assert str(native.value) == str(original.value)


def test_frobenius_norm_is_not_entrywise_or_spectral_tolerance() -> None:
    from qualtran.linalg.testing import assert_matrices_almost_equal as source

    left = np.zeros((2, 2))
    right = 0.08 * np.eye(2)
    np.testing.assert_allclose(left, right, atol=0.1, rtol=0)
    assert np.linalg.norm(left - right, ord=2) < 0.1
    for check in (source, assert_matrices_almost_equal):
        with pytest.raises(AssertionError):
            check(left, right, atol=0.1)


def test_nonfinite_and_nonunitary_source_boundaries() -> None:
    from qualtran.linalg.testing import assert_matrices_almost_equal as source_matrix
    from qualtran.linalg.testing import (
        assert_unitaries_equivalent_upto_global_phase as source_unitary,
    )

    identity = np.eye(2)
    for check in (source_matrix, assert_matrices_almost_equal):
        with pytest.raises(AssertionError):
            check(identity, identity * np.nan)
        with pytest.raises(AssertionError):
            check(identity, identity, atol=np.nan)
    # Source trace metric assumes unitarity; its clamp accepts this nonunitary input.
    for check in (source_unitary, assert_unitaries_equivalent_upto_global_phase):
        check(identity, 2 * identity)
        with pytest.raises(AssertionError):
            check(identity, identity, atol=-1)


def test_phaseful_append_consumer_and_checker_replacement() -> None:
    from qualtran.linalg.testing import assert_matrices_almost_equal as source_matrix
    from qualtran.linalg.testing import (
        assert_unitaries_equivalent_upto_global_phase as source_unitary,
    )

    matrix_type = NDArray[np.float32 | np.float64 | np.complex64 | np.complex128]

    def inspect(
        check: Callable[[matrix_type, matrix_type], None],
    ) -> NDArray[np.complex128]:
        recipe = Circuit(2, dtype=torch.complex128).h(0).rz(0, theta=0.31).to_ir()
        circuit = Circuit(2, dtype=torch.complex128).append(
            FieldOperation(2, recipe.instructions)
        )
        assert circuit.to_ir().instructions == recipe.instructions
        state: NDArray[np.complex128] = circuit.state().detach().numpy().reshape(2, 2)
        expected = np.zeros((2, 2), dtype=np.complex128)
        expected[:, 0] = np.exp(1j * np.array([-0.155, 0.155])) / np.sqrt(2)
        check(state, expected)
        np.testing.assert_array_equal(state[:, 1], 0)
        assert len(recipe.instructions) == 2
        return state

    np.testing.assert_array_equal(
        inspect(assert_matrices_almost_equal), inspect(source_matrix)
    )
    identity = np.eye(2, dtype=np.complex128)
    phaseful = np.exp(0.31j) * identity
    for check in (source_unitary, assert_unitaries_equivalent_upto_global_phase):
        check(identity, phaseful)
    for check in (source_matrix, assert_matrices_almost_equal):
        with pytest.raises(AssertionError):
            check(identity, phaseful)
    controlled = np.eye(4, dtype=np.complex128)
    controlled[2:, 2:] = phaseful
    for check in (source_unitary, assert_unitaries_equivalent_upto_global_phase):
        with pytest.raises(AssertionError):
            check(np.eye(4), controlled)
