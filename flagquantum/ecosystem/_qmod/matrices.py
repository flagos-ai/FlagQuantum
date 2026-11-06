"""Finite source matrix constants and unitary calls at the Qmod boundary."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import MAX_MATRIX_QUBITS
from .phase import Location
from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def matrix_value(node: Expression) -> complex:
    text = node.token.text
    if not node.children:
        return complex(float(text))
    left = matrix_value(node.children[0])
    if text == "j":
        return 1j * left
    right = matrix_value(node.children[1]) if len(node.children) == 2 else None
    if right is None:
        return -left if text == "-" else left
    return left - right if text == "-" else left + right


def validate_matrix_constant(token: Token, matrix: Expression) -> None:
    size = 1 << MAX_MATRIX_QUBITS
    if not 1 <= len(matrix.children) <= size or any(
        not 1 <= len(row.children) <= size for row in matrix.children
    ):
        reject(
            token, "unitary", f"matrix constants are bounded to {size} rows and columns"
        )
    if len({len(row.children) for row in matrix.children}) != 1:
        reject(token, "unitary", "matrix constants must be rectangular")
    for row in matrix.children:
        for item in row.children:
            value = matrix_value(item)
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                reject(item.token, "unitary", "matrix entries must be finite")


def lower_unitary(
    owner: _Adapt,
    token: Token,
    args: tuple[Expression, ...],
    *,
    selection: tuple[tuple[Location, ...], int] | None = None,
) -> None:
    if len(args) != 2:
        reject(token, "signature", "unitary requires a matrix and qubit array")
    matrix, target_arg = args
    if not matrix.children:
        matrix = owner.constants.get(matrix.token.text, matrix)
    if matrix.token.text != "[":
        reject(
            matrix.token,
            "unitary",
            "unitary requires a literal matrix or matrix constant",
        )
    view = (
        (owner.gate_target(target_arg),)
        if target_arg.token.text == "[" and not target_arg.children[0].children
        else owner.array_view(target_arg)
    )
    if not 1 <= len(view) <= MAX_MATRIX_QUBITS or len(set(view)) != len(view):
        reject(
            target_arg.token,
            "unitary",
            "unitary requires one to four distinct qubit locations",
        )
    view_attributes: dict[str, object] = {}
    if target_arg.children or target_arg.token.text in owner.views:
        # Qmod array position zero is the least-significant matrix bit.
        wires = tuple(reversed(view))
        operands = tuple(register for register, _ in wires)
        view_attributes["indices"] = tuple(index for _, index in wires)
    else:
        operands = (view[0][0],)
    entries = tuple(
        tuple((value.real, value.imag) for value in map(matrix_value, row.children))
        for row in matrix.children
    )
    if selection is not None:
        controls, expected = selection
        if len(set((*controls, *view))) != len(controls) + len(view):
            reject(
                token, "alias", "unitary targets must be disjoint from condition inputs"
            )
        if len(controls) + len(
            view
        ) > MAX_MATRIX_QUBITS or not 0 <= expected < 1 << len(controls):
            reject(
                token,
                "unitary",
                "selected unitary exceeds the matrix width or selector range",
            )
        size = 1 << len(view)
        if len(entries) != size or any(len(row) != size for row in entries):
            reject(token, "unitary", "unitary matrix must match the target width")
        expanded_size = size << len(controls)
        start = expected * size
        entries = tuple(
            tuple(
                (
                    entries[row - start][col - start]
                    if start <= row < start + size and start <= col < start + size
                    else (float(row == col), 0.0)
                )
                for col in range(expanded_size)
            )
            for row in range(expanded_size)
        )
        wires = (*reversed(controls), *reversed(view))
        operands = tuple(register for register, _ in wires)
        view_attributes["indices"] = tuple(index for _, index in wires)
    owner.effect_op("qreg.unitary", token, operands, matrix=entries, **view_attributes)
