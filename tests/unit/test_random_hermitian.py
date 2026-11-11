"""Sampling preserves the original matrix, RNG consumption, and failure order."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.simulation.numerics.random_hermitian import random_hermitian_matrix

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("seed", [0, 17, 1994])
def test_matrix_and_following_random_draws_match_source(
    dimension: int, seed: int
) -> None:
    from qualtran.linalg.matrix import random_hermitian_matrix as source

    native_rng, source_rng = np.random.RandomState(seed), np.random.RandomState(seed)
    for _ in range(2):
        actual = random_hermitian_matrix(dimension, native_rng)
        expected = source(dimension, source_rng)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (dimension, dimension) and actual.dtype == np.float64
        np.testing.assert_array_equal(actual, actual.conj().T)
        if dimension:
            assert np.linalg.norm(actual, ord=2) <= 1 + 2e-15
    np.testing.assert_array_equal(native_rng.randn(11), source_rng.randn(11))


def test_no_global_rng_consumption() -> None:
    before: object = np.random.get_state(legacy=True)
    random_hermitian_matrix(4, np.random.RandomState(11))
    after: object = np.random.get_state(legacy=True)
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_dimension_failure_matches_source_without_consuming_rng() -> None:
    from qualtran.linalg.matrix import random_hermitian_matrix as source

    native_rng, source_rng = np.random.RandomState(21), np.random.RandomState(21)
    with pytest.raises(ValueError) as original:
        source(-1, source_rng)
    with pytest.raises(ValueError) as native:
        random_hermitian_matrix(-1, native_rng)
    assert str(native.value) == str(original.value)
    np.testing.assert_array_equal(native_rng.randn(11), source_rng.randn(11))


def test_qr_failure_occurs_after_one_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    from qualtran.linalg.matrix import random_hermitian_matrix as source

    def refuse(
        matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert matrix.shape == (3, 3)
        raise np.linalg.LinAlgError("test QR failure")

    monkeypatch.setattr(np.linalg, "qr", refuse)
    native_rng, source_rng, expected_rng = (np.random.RandomState(32) for _ in range(3))
    for evaluator, rng in ((source, source_rng), (random_hermitian_matrix, native_rng)):
        with pytest.raises(np.linalg.LinAlgError, match="test QR failure"):
            evaluator(3, rng)
    expected_rng.randn(3, 3)
    expected = expected_rng.randn(11)
    np.testing.assert_array_equal(native_rng.randn(11), expected)
    np.testing.assert_array_equal(source_rng.randn(11), expected)


def test_degenerate_qr_is_not_silently_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg.matrix import random_hermitian_matrix as source

    def degenerate(
        matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.eye(matrix.shape[0]), np.zeros_like(matrix)

    monkeypatch.setattr(np.linalg, "qr", degenerate)
    for evaluator in (source, random_hermitian_matrix):
        with pytest.warns(RuntimeWarning, match="invalid value"):
            result = evaluator(2, np.random.RandomState(8))
        assert np.isnan(result).all()


def test_same_consumer_accepts_source_and_native_sampler() -> None:
    from qualtran.linalg.matrix import random_hermitian_matrix as source

    def evolve(
        sampler: Callable[[int, np.random.RandomState], NDArray[np.float64]],
    ) -> NDArray[np.complex128]:
        matrix = sampler(2, np.random.RandomState(109))
        values, vectors = np.linalg.eigh(matrix)
        evolution = (vectors * np.exp(-0.37j * values)) @ vectors.conj().T
        recipe = Circuit(2, dtype=torch.complex128).h(0).rz(0, theta=0.31).to_ir()
        circuit = Circuit(2, dtype=torch.complex128)
        circuit.append(FieldOperation(2, recipe.instructions))
        assert circuit.to_ir().instructions == recipe.instructions
        state = circuit.state().detach().numpy().reshape(2, 2)
        evolved: NDArray[np.complex128] = evolution @ state
        np.testing.assert_array_equal(evolved[:, 1], 0)
        assert len(recipe.instructions) == 2
        return evolved

    # Replacement changes no consumer code, including append and clean-wire checks.
    np.testing.assert_array_equal(evolve(random_hermitian_matrix), evolve(source))
