"""Adapt native phase syntax and predicate storage to shared Boolean factors."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn

from .. import _phase_factors
from .._hybrid.model import Operation, Value, ValueId
from .._phase_factors import Cube
from .._phase_polynomial import MAX_PHASE_INPUT_BITS, Location, Polynomial, expand_phase
from .frontend_python import _error
from .numeric_expression import WordExpression
from .register_profile import register_type, register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def expand_factored_phase(
    owner: _Body,
    expression: ast.expr,
    quantum_word: Callable[[ast.expr], tuple[Location, ...] | None],
    number: Callable[[ast.expr], float],
    operator: Callable[[ast.expr], str],
    children: Callable[[ast.expr], tuple[ast.expr, ...]],
    fail: Callable[[ast.expr, str, str], NoReturn],
) -> tuple[Polynomial, tuple[Location, ...], tuple[tuple[Cube, Value], ...]] | None:
    placeholders: dict[ast.expr, int] = {}

    def replace(node: ast.expr, operands: tuple[ast.expr, ...]) -> ast.expr:
        if isinstance(node, ast.BinOp):
            return ast.copy_location(
                ast.BinOp(left=operands[0], op=node.op, right=operands[1]), node
            )
        if isinstance(node, ast.UnaryOp):
            return ast.copy_location(ast.UnaryOp(op=node.op, operand=operands[0]), node)
        if isinstance(node, ast.Compare):
            return ast.copy_location(
                ast.Compare(
                    left=operands[0], ops=node.ops, comparators=list(operands[1:])
                ),
                node,
            )
        return node

    def marker(node: ast.expr, index: int) -> ast.expr:
        result = ast.copy_location(
            ast.Name(id=f"phase_factor_{index}", ctx=ast.Load()), node
        )
        placeholders[result] = index
        return result

    factored, factors = _phase_factors.preserve_factors(
        expression,
        quantum_word,
        operator_of=operator,
        operands_of=children,
        replace_operands=replace,
        make_factor=marker,
    )
    if not factors:
        return None
    values = tuple(
        Value(ValueId(owner.name, owner.counter + i), register_type(1))
        for i in range(len(factors))
    )

    def word(node: ast.expr) -> tuple[Location, ...] | None:
        if node in placeholders:
            return ((values[placeholders[node]], 0),)
        return quantum_word(node)

    polynomial, reads = expand_phase(
        factored, word, number, operator_of=operator, operands_of=children, fail=fail
    )
    physical = {location for factor in factors for location, _ in factor} | {
        location for location in reads if location[0] not in values
    }
    if len(physical) > MAX_PHASE_INPUT_BITS:
        fail(expression, "budget.phase", "phase exceeds 24 quantum input bits")
    owner.counter += len(factors)
    return polynomial, reads, tuple(zip(factors, values, strict=True))


def emit_factor(owner: _Body, statement: ast.Call, factor: Cube, value: Value) -> None:
    owners = tuple(dict.fromkeys(register for (register, _), _ in factor))
    widths = {
        f"owner{index}": register_width(register)
        for index, register in enumerate(owners)
    }
    predicates: list[ast.expr] = []
    for index, register in enumerate(owners):
        mask = sum(1 << bit for (source, bit), _ in factor if source == register)
        expected = sum(
            1 << bit
            for (source, bit), positive in factor
            if source == register and positive
        )
        predicates.append(
            ast.Compare(
                left=ast.BinOp(
                    left=ast.Name(id=f"owner{index}", ctx=ast.Load()),
                    op=ast.BitAnd(),
                    right=ast.Constant(value=mask),
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=expected)],
            )
        )
    expression = predicates[0]
    for predicate in predicates[1:]:
        expression = ast.BinOp(left=expression, op=ast.BitAnd(), right=predicate)

    def text(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        return "==" if isinstance(node, ast.Compare) else "&"

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.Compare):
            return node.left, node.comparators[0]
        return ()

    def fail(node: ast.expr, code: str, message: str) -> NoReturn:
        _error(owner.source, statement, f"program.phase.factor.{code}", message)

    graph = WordExpression(
        expression,
        widths,
        f"{owner.name}.factor{value.id.index}",
        set(),
        text=text,
        children=children,
        location=lambda node: owner.source.location(statement),
        reject=fail,
    ).function()
    owner.owner.functions.append(graph)
    owner.operations.append(
        Operation(
            "qreg.local",
            results=(value,),
            attributes={"name": graph.name},
            location=owner.source.location(statement),
        )
    )
    owner.emit("qreg.compute", statement, (value, *owners), callee=graph.name)
