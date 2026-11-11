"""Keep bounded integer callback controls on existing quantum storage."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from .._phase_polynomial import Location
from .frontend_phase import capture_polynomial_phase
from .frontend_python import _error
from .frontend_views import gate_target, locations

if TYPE_CHECKING:
    from .frontend_program import _Body


def scalar_comparison(owner: _Body, condition: ast.expr) -> tuple[ast.expr, bool]:
    """Recognize exact raw one-bit equality without manufacturing a result bit."""
    if not (
        isinstance(condition, ast.Compare)
        and len(condition.ops) == 1
        and isinstance(condition.ops[0], (ast.Eq, ast.NotEq))
        and len(condition.comparators) == 1
    ):
        return condition, False
    left, right = condition.left, condition.comparators[0]
    if isinstance(left, ast.Constant):
        left, right = right, left
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    if not (
        isinstance(left, ast.Name)
        and left.id in ports
        and isinstance(right, ast.Constant)
        and type(right.value) is int
        and right.value in (0, 1)
    ):
        return condition, False
    port = ports[left.id]
    if port.kind not in {"qbit", "qnum"} or port.signed or port.fractional_bits:
        return condition, False
    view = locations(owner, left)
    if len(view) != 1:
        return condition, False
    negative = (right.value == 0) != isinstance(condition.ops[0], ast.NotEq)
    if negative and (
        port.role not in {"inout", "output"}
        or any(
            candidate.role == "const" and owner.registers.get(name) == view[0][0]
            for name, candidate in owner.ports.items()
        )
    ):
        return condition, False
    return left, negative


def capture_predicate_x(
    owner: _Body, condition: ast.expr, callback: ast.Call, statement: ast.Call
) -> bool:
    """Use H Z H on arbitrary markers; retain the previous path outside bounds."""
    if not owner.owner.expanded_callbacks or len(callback.args) != 1:
        return False
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    allowed = (
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.UnaryOp,
        ast.USub,
    )
    for node in ast.walk(condition):
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return False
            for operand in (node.left, node.comparators[0]):
                if any(not isinstance(child, allowed) for child in ast.walk(operand)):
                    return False
        if isinstance(node, ast.Constant) and type(node.value) is not int:
            return False
    reads: set[Location] = set()
    for child in ast.walk(condition):
        if not isinstance(child, ast.Name):
            continue
        if child.id not in ports:
            value = owner.phase_number(child)
            if value is None or not value.is_integer():
                return False
            continue
        if ports[child.id].fractional_bits:
            return False
        reads.update(locations(owner, child))
    if len(reads) > 8:
        return False
    target, bit = gate_target(owner, callback.args[0])
    if any(value == target for value, _ in reads):
        _error(
            owner.source,
            statement,
            "program.control",
            "condition and target must be disjoint",
        )
    owner.emit("qreg.gate", statement, (target,), gate="h", index=bit)
    capture_polynomial_phase(
        owner, callback.args[0], math.pi, statement, condition=condition
    )
    owner.emit("qreg.gate", statement, (target,), gate="h", index=bit)
    return True
