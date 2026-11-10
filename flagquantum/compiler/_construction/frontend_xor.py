"""Capture reversible expression XOR without fresh-output or zero assumptions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .frontend_numeric import has_numeric
from .frontend_python import _error

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_expression_xor(owner: _Body, node: ast.AugAssign) -> None:
    if (
        owner.aliases is not None
        or not isinstance(node.target, ast.Name)
        or node.target.id not in owner.ports
        or owner.ports[node.target.id].kind not in {"qnum", "qbit"}
        or owner.ports[node.target.id].role != "output"
        or has_numeric(owner.ports[node.target.id])
    ):
        _error(
            owner.source,
            node,
            "program.xor",
            "expression XOR requires a root-owned unsigned scalar Output",
        )
    name = node.target.id
    if any(
        isinstance(child, ast.Name) and child.id == name
        for child in ast.walk(node.value)
    ):
        _error(
            owner.source,
            node,
            "program.xor.alias",
            "expression XOR target cannot alias an input",
        )
    destination = owner.register(node.target)
    expression, inputs = owner.capture_expression(node.value, node)
    owner.emit("qreg.xor", node, (destination, *inputs), callee=expression.name)
