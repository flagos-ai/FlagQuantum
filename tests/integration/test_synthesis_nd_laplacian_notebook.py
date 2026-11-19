"""The complete periodic Laplacian workflow preserves grid, phase and block scale."""

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np

from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.overview.model import (
    laplacian_1d,
    laplacian_nd,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.overview.workflow import (
    run_case,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("size", (2, 4, 8, 32))
def test_fresh_periodic_matrix_wraps_with_exact_scale(size: int) -> None:
    matrix = laplacian_1d(size)
    expected = np.zeros((size, size), dtype=complex)
    for column in range(size):
        expected[column, column] -= 0.5
        expected[(column - 1) % size, column] += 0.25
        expected[(column + 1) % size, column] += 0.25
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_allclose(matrix.sum(axis=0), 0)
    np.testing.assert_array_equal(matrix, matrix.conj().T)


@pytest.mark.parametrize("sizes", ((2, 4), (4, 2), (2, 2, 2)))
def test_reversed_axis_tensor_layout_and_selector_power_of_two_scale(
    sizes: tuple[int, ...],
) -> None:
    matrix = laplacian_nd(sizes)
    total = int(np.prod(sizes))
    expected = np.zeros((total, total), dtype=complex)
    divisor = 1 << (len(sizes) - 1).bit_length()
    for column in range(total):
        stride = 1
        for size in sizes:
            coordinate = (column // stride) % size
            expected[column, column] -= 0.5 / divisor
            expected[
                column + (((coordinate - 1) % size) - coordinate) * stride, column
            ] += (0.25 / divisor)
            expected[
                column + (((coordinate + 1) % size) - coordinate) * stride, column
            ] += (0.25 / divisor)
            stride *= size
    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.parametrize("multidimensional", (False, True))
def test_source_program_matches_absolute_full_state_and_raw_projected_column(
    multidimensional: bool,
) -> None:
    result = run_case(multidimensional=multidimensional)
    assert result.resources.n_qubits == (6 if multidimensional else 7)
    assert result.full_state_error < 1e-11
    assert result.block_error < 1e-11
    assert result.projected[0].real == pytest.approx(-0.5, abs=1e-12)
    assert result.source_display[0] == pytest.approx(0.5, abs=1e-12)
    assert result.projection_probability == pytest.approx(
        0.34375 if multidimensional else 0.375, abs=1e-12
    )
    assert result.fidelity == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(result.projected, result.expected, atol=1e-11, rtol=0)
    assert result.program.selection.objective == "depth"


@pytest.mark.parametrize("size", (True, 0, 3, 64))
def test_invalid_grid_refuses_before_matrix_construction(size: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        laplacian_1d(size)


@pytest.mark.parametrize("sizes", ((), (2, 2, 2, 2, 2), (32, 32)))
def test_oversized_reference_refuses(sizes: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="dimensions|256"):
        laplacian_nd(sizes)


def test_all_seven_source_stages_are_mapped_without_executed_outputs() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/paper_implementation_project/block_encoding_nd_laplacian/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        5,
        6,
        7,
        8,
        11,
        12,
        13,
    )
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
