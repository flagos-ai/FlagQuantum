from __future__ import annotations

import importlib
import itertools
import os
from typing import Any

import pytest
import torch

from flagquantum.ecosystem.kaiwu import (
    encode_qubo_as_ising,
    ising_energy,
)

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.integration
SOURCE_CONFORMANCE = "FLAGQUANTUM_TEST_KAIWU_SOURCE"


def _require_source_conformance() -> Any:
    if os.environ.get(SOURCE_CONFORMANCE) != "1":
        pytest.skip(f"set {SOURCE_CONFORMANCE}=1 through the pinned source runner")
    return importlib.import_module("kaiwu.core")


def test_energy_convention_matches_kaiwu_community() -> None:
    kaiwu_core = _require_source_conformance()
    matrix = np.array([[0.5, 1.25, -0.5], [1.25, -1.0, 0.75], [-0.5, 0.75, 2.0]])
    solutions = np.array(list(itertools.product((-1, 1), repeat=3)))

    _, expected = kaiwu_core.get_sorted_solutions(
        matrix,
        solutions.copy(),
        bias=3.5,
        negtail_ff=False,
        sort_solutions=False,
    )
    actual = ising_energy(matrix.tolist(), solutions.tolist(), bias=3.5)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=0.0)


def test_qubo_encoding_matches_kaiwu_community_gauge() -> None:
    kaiwu_core = _require_source_conformance()
    qubo = np.array([[1.5, -2.0, 0.25], [-2.0, 3.0, 1.0], [0.25, 1.0, -0.5]])

    expected_matrix, expected_bias = kaiwu_core.qubo_matrix_to_ising_matrix(qubo.copy())
    actual = encode_qubo_as_ising(qubo.tolist())

    np.testing.assert_allclose(actual.matrix.numpy(), expected_matrix)
    assert actual.bias == pytest.approx(expected_bias)


class _FlagQuantumExactFake:
    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        solutions = torch.tensor(
            list(itertools.product((-1, 1), repeat=ising_matrix.shape[0]))
        )
        energies = ising_energy(ising_matrix.tolist(), solutions)
        return solutions[energies == energies.min()].numpy()


def _minimum_energy(solver: Any, matrix: np.ndarray) -> float:
    solutions = solver.solve(matrix.copy())
    energies = ising_energy(matrix.tolist(), solutions.tolist())
    return float(energies.min().item())


def test_fake_and_kaiwu_solver_are_replaceable_for_consumer() -> None:
    kaiwu_core = _require_source_conformance()

    class ExhaustiveKaiwuSolver(kaiwu_core.IsingSolver):
        def _solve(self, ising_matrix: np.ndarray | None = None) -> np.ndarray:
            assert ising_matrix is not None
            return np.array(
                list(itertools.product((-1, 1), repeat=ising_matrix.shape[0]))
            )

    matrix = np.array([[0.0, 2.0, -1.0], [2.0, 0.0, 0.5], [-1.0, 0.5, 0.0]])

    fake_energy = _minimum_energy(_FlagQuantumExactFake(), matrix)
    kaiwu_energy = _minimum_energy(ExhaustiveKaiwuSolver(), matrix)

    assert fake_energy == kaiwu_energy
