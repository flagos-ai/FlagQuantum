"""Finite source matrix constants and unitary calls at the Qmod boundary."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
    if not 1 <= len(matrix.children) <= 8 or any(
        not 1 <= len(row.children) <= 8 for row in matrix.children
    ):
        reject(
            token, "unitary", "matrix constants are bounded to eight rows and columns"
        )
    if len({len(row.children) for row in matrix.children}) != 1:
        reject(token, "unitary", "matrix constants must be rectangular")
    for row in matrix.children:
        for item in row.children:
            value = matrix_value(item)
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                reject(item.token, "unitary", "matrix entries must be finite")


def lower_unitary(owner: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
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
    view = owner.array_view(target_arg)
    if len(view) not in {1, 2, 3} or len(set(view)) != len(view):
        reject(
            target_arg.token,
            "unitary",
            "unitary requires one to three distinct qubit locations",
        )
    view_attributes: dict[str, object] = {}
    if target_arg.children or target_arg.token.text in owner.views:
        # Qmod array position zero is the least-significant matrix bit.
        wires = tuple(reversed(view))
        operands = tuple(register for register, _ in wires)
        view_attributes["indices"] = tuple(index for _, index in wires)
    else:
        operands = (view[0][0],)
    owner.effect_op(
        "qreg.unitary",
        token,
        operands,
        matrix=tuple(
            tuple((value.real, value.imag) for value in map(matrix_value, row.children))
            for row in matrix.children
        ),
        **view_attributes,
    )
