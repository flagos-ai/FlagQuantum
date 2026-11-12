"""Bind native fixed-point XOR expressions to shared exact Compiler semantics."""

from __future__ import annotations

import ast
from fractions import Fraction
from typing import TYPE_CHECKING, NoReturn

from .._fixed_word import exact_expression, fixed_xor_table, truth_cubes
from .._phase_polynomial import Location
from .frontend_python import _error, _QuantumFunction
from .frontend_views import locations
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def is_fixed_xor_helper(function: object) -> bool:
    """Recognize the one-update borrowing profile without executing user code."""
    if not isinstance(function, _QuantumFunction) or not function.source.program:
        return False
    body = function.source.node.body
    return (
        len(body) == 1
        and isinstance(body[0], ast.AugAssign)
        and isinstance(body[0].op, ast.BitXor)
        and isinstance(body[0].target, ast.Name)
    )


def capture_fixed_xor(owner: _Body, node: ast.AugAssign) -> bool:
    """Admit whole unsigned fractional words, preserving physical formal aliases."""
    if not isinstance(node.target, ast.Name):
        return False
    visible = owner.alias_ports if owner.aliases is not None else owner.ports
    assert visible is not None
    if node.target.id not in visible:
        return False
    if owner.aliases is None and not visible[node.target.id].fractional_bits:
        return False
    target_view = locations(owner, node.target)
    destination = target_view[0][0]
    target_port = next(
        owner.ports[label]
        for label, value in owner.registers.items()
        if value == destination
    )
    if not target_port.fractional_bits:
        return False

    def fail(expression: ast.expr, message: str) -> NoReturn:
        _error(owner.source, expression, "program.fixed_xor", message)

    if (
        target_port.signed
        or target_port.kind != "qnum"
        or visible[node.target.id].role not in {"output", "inout"}
        or target_port.role == "const"
        or target_view
        != tuple((destination, bit) for bit in range(register_width(destination)))
    ):
        fail(node.target, "fixed-point XOR requires a whole writable unsigned scalar")
    operands: dict[str, tuple[tuple[Location, ...], int]] = {}

    def name(expression: ast.expr) -> str | None:
        if not isinstance(expression, ast.Name) or expression.id not in visible:
            return None
        port = visible[expression.id]
        view = locations(owner, expression)
        # Resolving a view may declare a previously unseen input register.
        actual = next(
            owner.ports[label]
            for label, value in owner.registers.items()
            if value == view[0][0]
        )
        if port.kind not in {"qnum", "qbit"} or actual.signed:
            fail(expression, "fixed-point XOR reads require unsigned scalar words")
        if view != tuple(
            (view[0][0], bit) for bit in range(register_width(view[0][0]))
        ):
            fail(expression, "fixed-point XOR inputs require complete scalar words")
        operands[expression.id] = view, actual.fractional_bits
        return expression.id

    def constant(expression: ast.expr) -> Fraction | None:
        if isinstance(expression, (ast.IfExp, ast.Compare)):
            return None
        value = owner.phase_number(expression)
        return None if value is None else Fraction(str(value))

    def operator(expression: ast.expr) -> str:
        if isinstance(expression, ast.IfExp):
            return "select"
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                fail(expression, "fixed-point selection requires one comparison")
            comparisons = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            return comparisons.get(type(expression.ops[0]), "")
        if isinstance(expression, (ast.BinOp, ast.UnaryOp)):
            for kind, symbol in (
                (ast.Add, "+"),
                (ast.Sub, "-"),
                (ast.Mult, "*"),
                (ast.USub, "-"),
            ):
                if isinstance(expression.op, kind):
                    return symbol
        return ""

    def children(expression: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(expression, ast.IfExp):
            return expression.test, expression.body, expression.orelse
        if isinstance(expression, ast.Compare):
            return expression.left, *expression.comparators
        if isinstance(expression, ast.BinOp):
            return expression.left, expression.right
        if isinstance(expression, ast.UnaryOp):
            return (expression.operand,)
        return ()

    evaluate = exact_expression(
        node.value,
        name=name,
        constant=constant,
        operator=operator,
        children=children,
        fail=fail,
    )
    try:
        reads, table = fixed_xor_table(
            operands,
            target_view,
            target_fraction=target_port.fractional_bits,
            evaluate=evaluate,
        )
    except ValueError as error:
        fail(node.value, str(error))
    # Witness target and reads even when the entire update cancels.
    for value, bit in (*reads, *target_view):
        owner.emit("qreg.control_phase", node, (value,), indices=(bit,), angle=0.0)
    for bit, target in enumerate(target_view):
        for cube in truth_cubes(tuple(word >> bit & 1 for word in table), len(reads)):
            selected = tuple(reads[index] for index, _ in cube)
            if not selected:
                owner.emit("qreg.gate", node, (target[0],), gate="x", index=target[1])
            else:
                combined = (*selected, target)
                owner.emit(
                    "qreg.control_x",
                    node,
                    tuple(value for value, _ in combined),
                    indices=tuple(index for _, index in combined),
                    control_state=sum(value << i for i, (_, value) in enumerate(cube)),
                )
    return True
