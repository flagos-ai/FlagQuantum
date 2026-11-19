"""Capture reversible expression XOR without fresh-output or zero assumptions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..verification.register_profile import register_width
from .fixed_xor import capture_fixed_xor
from .numeric import has_numeric, has_numeric_reads, reject_numeric_reads
from .phase import expand_native_phase
from .python import _error
from .views import locations

if TYPE_CHECKING:
    from .program import _Body


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
    if _capture_boolean_xor(owner, node):
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


def _capture_boolean_xor(owner: _Body, node: ast.AugAssign) -> bool:
    """Toggle a scalar bit from bounded Boolean reads of static quantum views."""
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    if (
        not isinstance(node.target, ast.Name)
        or node.target.id not in ports
        or ports[node.target.id].kind != "qbit"
    ):
        return False

    def boolean(expression: ast.expr) -> bool:
        if isinstance(expression, ast.Constant):
            return type(expression.value) is int and expression.value in (0, 1)
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            return boolean(expression.left) and boolean(expression.right)
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, ast.Invert
        ):
            return boolean(expression.operand)
        if isinstance(expression, (ast.Name, ast.Subscript)):
            base: ast.expr = expression
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name) and base.id in ports:
                return len(locations(owner, expression)) == 1
        return False

    if not boolean(node.value):
        return False
    if ports[node.target.id].role not in {"output", "inout"}:
        _error(owner.source, node, "program.xor", "Boolean XOR requires a writable bit")
    target = locations(owner, node.target)
    if len(target) != 1:
        _error(
            owner.source,
            node.target,
            "program.xor",
            "Boolean XOR requires one target bit",
        )
    expanded = expand_native_phase(
        owner,
        ast.Constant(value=1),
        1.0,
        node,
        condition=node.value,
    )
    assert expanded is not None
    polynomial, reads, factors = expanded
    assert not factors
    if target[0] in reads:
        _error(
            owner.source,
            node,
            "program.xor.alias",
            "Boolean XOR target cannot alias an input",
        )
    # Keep all reads visible to ownership checks, including canceled terms.
    for value, bit in (*reads, *target):
        owner.emit("qreg.control_phase", node, (value,), indices=(bit,), angle=0.0)
    for mask, coefficient in sorted(polynomial.items()):
        if int(coefficient) % 2 == 0:
            continue
        controls = tuple(
            location for index, location in enumerate(reads) if mask & (1 << index)
        )
        if not controls:
            owner.emit("qreg.gate", node, (target[0][0],), gate="x", index=target[0][1])
        else:
            combined = (*controls, *target)
            owner.emit(
                "qreg.control_x",
                node,
                tuple(value for value, _ in combined),
                indices=tuple(bit for _, bit in combined),
            )
    return True
