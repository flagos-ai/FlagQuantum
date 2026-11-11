"""Capture reversible expression XOR without fresh-output or zero assumptions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .frontend_fixed_xor import capture_fixed_xor
from .frontend_numeric import has_numeric, has_numeric_reads, reject_numeric_reads
from .frontend_python import _error
from .frontend_views import locations
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def _predicate_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.Constant):
        return type(node.value) is int and node.value in (0, 1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return _predicate_expression(node.operand)
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
        and _predicate_expression(node.left)
        and _predicate_expression(node.right)
    )


def capture_expression_xor(owner: _Body, node: ast.AugAssign) -> None:
    if capture_fixed_xor(owner, node):
        return
    if has_numeric_reads(owner, node.value) and not _predicate_expression(node.value):
        reject_numeric_reads(owner, node.value, "non-predicate XOR expressions")
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    if (
        not isinstance(node.target, ast.Name)
        or node.target.id not in ports
        or ports[node.target.id].kind not in {"qnum", "qbit"}
        or ports[node.target.id].role
        not in ({"output", "inout"} if owner.aliases is None else {"inout"})
        or has_numeric(ports[node.target.id])
    ):
        _error(
            owner.source,
            node,
            "program.xor",
            "expression XOR requires an unsigned scalar Output or borrowed inout register",
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
    if owner.aliases is None:
        destination = owner.register(node.target)
    else:
        view = locations(owner, node.target)
        destination = view[0][0]
        if view != tuple(
            (destination, bit) for bit in range(register_width(destination))
        ):
            _error(
                owner.source,
                node.target,
                "program.borrow",
                "expression XOR target must borrow a complete raw register in logical bit order",
            )
    expression, inputs = owner.capture_expression(node.value, node)
    owner.emit("qreg.xor", node, (destination, *inputs), callee=expression.name)
