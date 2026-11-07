"""Static literal real matrix bindings and lexically destructured row traversal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .syntax import Expression, Statement, reject

if TYPE_CHECKING:
    from .context import _Adapt


def bound_matrix(owner: _Adapt, node: Expression) -> tuple[tuple[float, ...], ...]:
    if node.token.text == "$real_matrix":
        if not 1 <= len(node.children) <= 256:
            reject(node.token, "binding", "real matrix needs 1..256 rows")
        return tuple(owner.real_array(row) for row in node.children)
    source = node.children[0] if node.token.text == ":" else node
    matrix = owner.classical_matrices.get(source.token.text)
    if source.children or matrix is None:
        reject(source.token, "binding", "expected a bound real matrix")
    if node.token.text != ":":
        return matrix
    start, stop = (owner.static_integer(value) for value in node.children[1:])
    if not 0 <= start <= stop <= len(matrix):
        reject(node.token, "binding", "real matrix row slice is out of bounds")
    return matrix[start:stop]


def matrix_cell(owner: _Adapt, node: Expression) -> float | None:
    row_node, column_node = node.children
    if row_node.token.text != "[":
        return None
    source, row_index = row_node.children
    if source.children or source.token.text not in owner.classical_matrices:
        return None
    matrix = owner.classical_matrices[source.token.text]
    row = owner.static_integer(row_index)
    column = owner.static_integer(column_node)
    if not 0 <= row < len(matrix) or not 0 <= column < len(matrix[row]):
        reject(node.token, "binding", "real matrix cell is out of bounds")
    return matrix[row][column]


def lower_matrix_iteration(
    owner: _Adapt, statement: Statement, index: str, max_iterations: int
) -> None:
    variables = statement.matrix_row
    assert variables is not None
    names = tuple(item.text for item in variables)
    if len(set(names)) != len(names) or any(
        name in owner.parameters
        or name in owner.constants
        or name in owner.loop_indices
        or name in owner.classical_values
        or name in owner.definitions
        for name in names
    ):
        reject(
            statement.token, "name", "real matrix row names must be distinct and fresh"
        )
    matrix = bound_matrix(owner, statement.arguments[0])
    if any(len(row) != len(names) for row in matrix):
        reject(
            statement.token,
            "binding",
            "real matrix row width disagrees with its bindings",
        )
    owner.loop_iterations += len(matrix)
    if owner.loop_iterations > max_iterations:
        reject(
            statement.token,
            "budget.expansion",
            f"static loops exceed {max_iterations} iterations",
        )
    try:
        for row_index, row in enumerate(matrix):
            owner.classical_values.update(zip(names, row, strict=True))
            for ordinal, child in enumerate(statement.repeat_body):
                owner.statement(child, f"{index}.{row_index}.{ordinal}")
    finally:
        for name in names:
            owner.classical_values.pop(name, None)
