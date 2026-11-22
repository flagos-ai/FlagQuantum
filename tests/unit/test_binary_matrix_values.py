"""Check complete matrix values and their existing circuit synthesis consumers."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.binary_linear_map import (
    binary_linear_map_ir,
)
from flagquantum.core._binary_matrix import BinaryMatrix

pytestmark = pytest.mark.unit
_FIXTURE = Path(__file__).with_name("fixtures") / "binary_matrix_values.json"


def _packed(matrix: BinaryMatrix) -> str:
    return f"{matrix.rows},{matrix.columns}:" + "".join(
        f"{row:X}," for row in matrix.packed_rows
    )


def _observe(matrix: BinaryMatrix, other: BinaryMatrix, vector: int) -> str:
    factors = matrix.plu()
    parts = [
        str(matrix.rank()),
        str(matrix.row_words),
        str(matrix).encode().hex(),
        _packed(matrix.transposed()),
        _packed(matrix.inverse()),
        (
            ";".join(_packed(factor) for factor in factors)
            if factors is not None
            else "NONE"
        ),
    ]
    try:
        parts.append(_packed(matrix.multiply(other)))
    except ValueError:
        parts.append("ERROR")
    try:
        parts.append(f"0x{matrix.apply(vector):X}")
    except ValueError:
        parts.append("ERROR")
    edits = []
    if matrix.rows and matrix.columns:
        edited = matrix.set(0, matrix.columns - 1, True)
        edits.append(_packed(edited))
        edited = edited.set(0, matrix.columns - 1, False)
        edits.append(_packed(edited))
        edited = edited.xor_bit(matrix.rows - 1, 0)
        edits.append(_packed(edited))
        for first, second in ((0, 0), (0, matrix.rows - 1)):
            edited = edited.swap_rows(first, second)
            edits.append(_packed(edited))
        for source, target in ((0, matrix.rows - 1), (0, 0)):
            edited = edited.xor_row_into(source, target)
            edits.append(_packed(edited))
        edits.append(str(int(edited.packed_rows[0] == 0)))
    parts += [";".join(edits), "1", str(int(matrix != other))]
    return "|".join(parts)


def test_complete_original_matrix_outputs_and_failure_sentinels() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    assert {source["parent"] for source in fixture["sources"]} >= {
        "K-0306",
        "K-0307",
        "K-0308",
    }
    for case in fixture["cases"]:
        matrix = BinaryMatrix(
            tuple(int(row, 16) for row in case["rows"]), case["columns"]
        )
        other = BinaryMatrix(
            tuple(int(row, 16) for row in case["other_rows"]), case["other_columns"]
        )
        observed = _observe(matrix, other, int(case["vector"], 16))
        assert hashlib.sha256(observed.encode()).hexdigest() == case["digest"], case
        factors = matrix.plu()
        if factors is not None:
            permutation, lower, upper = factors
            assert permutation.multiply(lower.multiply(upper)) == matrix
        inverse = matrix.inverse()
        if inverse.rows:
            assert matrix.multiply(inverse) == BinaryMatrix.identity(matrix.rows)
            assert inverse.multiply(matrix) == BinaryMatrix.identity(matrix.rows)


@pytest.mark.parametrize("rows,columns", [((1,), 0), ((-1,), 3), ((1,), -1)])
def test_invalid_packed_values_are_refused(rows: tuple[int, ...], columns: int) -> None:
    with pytest.raises(ValueError):
        BinaryMatrix(rows, columns)


def test_dense_views_and_empty_shapes_retain_dimensions() -> None:
    examples: tuple[list[list[int]], ...] = ([], [[]], [[1, 0, 1], [0, 1, 0]])
    for dense in examples:
        matrix = BinaryMatrix.from_dense(dense)
        assert matrix.dense() == dense
        assert matrix.transposed().transposed() == matrix
    for dense in ([[1], []], [[2]]):
        with pytest.raises(ValueError):
            BinaryMatrix.from_dense(dense)
    assert BinaryMatrix((), 7) != BinaryMatrix()
    assert BinaryMatrix((0, 0), 0) != BinaryMatrix()
    assert BinaryMatrix.identity(0).plu() == (BinaryMatrix(),) * 3


def test_invalid_coordinates_fail_without_mutating_values() -> None:
    matrix = BinaryMatrix.identity(2)
    for row, column in ((-1, 0), (0, -1), (2, 1), (1, 2)):
        with pytest.raises(ValueError):
            matrix.get(row, column)
        with pytest.raises(ValueError):
            matrix.set(row, column, True)
        with pytest.raises(ValueError):
            matrix.xor_bit(row, column)
    for first, second in ((-1, 0), (0, 2)):
        with pytest.raises(ValueError):
            matrix.swap_rows(first, second)
        with pytest.raises(ValueError):
            matrix.xor_row_into(first, second)
    for value in (-1, 1 << 1024):
        with pytest.raises(ValueError):
            matrix.apply(value)
    assert matrix == BinaryMatrix.identity(2)


def test_append_and_exact_inverse_use_shared_matrix_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = BinaryMatrix((9, 4, 5, 6), 4)
    assert matrix.rank() == 4
    columns = matrix.transposed().packed_rows
    calls: list[str] = []
    original_plu, original_inverse = BinaryMatrix.plu, BinaryMatrix.inverse

    def record_plu(
        value: BinaryMatrix,
    ) -> tuple[BinaryMatrix, BinaryMatrix, BinaryMatrix] | None:
        calls.append("plu")
        return original_plu(value)

    def record_inverse(value: BinaryMatrix) -> BinaryMatrix:
        calls.append("inverse")
        return original_inverse(value)

    monkeypatch.setattr(BinaryMatrix, "plu", record_plu)
    monkeypatch.setattr(BinaryMatrix, "inverse", record_inverse)
    forward = binary_linear_map_ir(tuple(range(4)), columns)
    adjoint = binary_linear_map_ir(tuple(range(4)), columns, adjoint=True)
    assert calls == ["plu", "inverse", "plu"]
    indices = torch.arange(32, dtype=torch.float64)
    state = torch.complex(indices + 1, 5 - 2 * indices)
    state /= state.norm()
    expected = torch.empty_like(state)
    for index in range(32):
        expected[(matrix.apply(index >> 1) << 1) | (index & 1)] = state[index]
    circuit = Circuit(5, inputs=state, dtype=torch.complex128)
    actual = circuit.append(FieldOperation(5, forward)).state().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    restored = circuit.append(FieldOperation(5, adjoint)).state().flatten()
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    assert all(gate.name in {"cx", "swap"} for gate in forward + adjoint)
