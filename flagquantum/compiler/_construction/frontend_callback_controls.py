"""Keep bounded integer callback controls on existing quantum storage."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._phase_polynomial import MAX_PHASE_DEGREE, Location
from .boolean_lowering import BooleanCandidateError, _disjoint_cubes
from .frontend_phase import expand_native_phase
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
    """Emit checked Boolean XOR on arbitrary markers within existing bounds."""
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
    # The XOR target and enclosing coherent controls also occupy phase inputs.
    # Reserve one slot per expanded callback before choosing the scratch-free
    # path; callbacks without a new control may conservatively retain computation.
    if len(reads) + 1 + owner.owner.expanded_callbacks > MAX_PHASE_DEGREE:
        return False
    target, bit = gate_target(owner, callback.args[0])
    if any(value == target for value, _ in reads):
        _error(
            owner.source,
            statement,
            "program.control",
            "condition and target must be disjoint",
        )
    expanded = expand_native_phase(
        owner, ast.Constant(value=1), 1.0, statement, condition=condition
    )
    if expanded is None:
        _error(owner.source, statement, "program.control", "expected bounded predicate")
    polynomial, ordered_reads, factors = expanded
    if factors:
        _error(owner.source, statement, "program.control", "expected bounded predicate")
    table = [polynomial.get(mask, 0.0) for mask in range(1 << len(ordered_reads))]
    for shift in range(len(ordered_reads)):
        for mask in range(len(table)):
            if mask & (1 << shift):
                table[mask] += table[mask ^ (1 << shift)]
    if any(value not in (0.0, 1.0) for value in table):
        _error(owner.source, statement, "program.control", "expected Boolean predicate")
    try:
        cubes = _disjoint_cubes(
            tuple(index for index, value in enumerate(table) if value),
            len(ordered_reads),
        )
    except BooleanCandidateError:
        # Integer ANF coefficients determine XOR parity exactly.
        cubes = tuple(
            (mask, mask)
            for mask, weight in sorted(polynomial.items())
            if int(weight) % 2
        )
    covered = 0
    for mask, _ in cubes:
        covered |= mask
    # A false predicate must still check a writable, initialized marker.
    owner.emit("qreg.gate", statement, (target,), gate="i", index=bit)
    for index, (read_value, read_bit) in enumerate(ordered_reads):
        if not covered & (1 << index):
            owner.emit(
                "qreg.control_phase",
                statement,
                (read_value,),
                indices=(read_bit,),
                angle=0.0,
            )
    for mask, pattern in cubes:
        selected = tuple(
            (index, location)
            for index, location in enumerate(ordered_reads)
            if mask & (1 << index)
        )
        if not selected:
            owner.emit("qreg.gate", statement, (target,), gate="x", index=bit)
            continue
        state = sum(
            int(bool(pattern & (1 << index))) << ordinal
            for ordinal, (index, _) in enumerate(selected)
        )
        attributes: dict[str, object] = {}
        if state != (1 << len(selected)) - 1:
            attributes["control_state"] = state
        owner.emit(
            "qreg.control_x",
            statement,
            (*(value for _, (value, _) in selected), target),
            indices=(*(wire for _, (_, wire) in selected), bit),
            **attributes,
        )
    return True
