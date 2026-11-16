"""Bind finite classical values without executing captured Python functions."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .frontend_ports import _resolve, classical_parameters
from .frontend_python import _error, _QuantumFunction

if TYPE_CHECKING:
    from .frontend_program import _Body

ClassicalValue = int | float | tuple[int | float, ...]


def _checked_real(
    definition: _QuantumFunction, node: ast.expr | ast.stmt, value: object
) -> float:
    if (
        type(value) not in (int, float)
        or not isinstance(value, (int, float))
        or abs(value) > 2**53
        or not math.isfinite(value)
    ):
        _error(
            definition.source,
            node,
            "program.binding",
            "classical values must be finite reals in ±2**53 (excluding bool)",
        )
    return float(value)


def _checked_integer(
    definition: _QuantumFunction, node: ast.expr | ast.stmt, value: object
) -> int:
    if type(value) is not int or not isinstance(value, int) or abs(value) > 2**53:
        _error(
            definition.source,
            node,
            "program.binding",
            "integer bindings require built-in int in ±2**53, excluding bool and float",
        )
    return value


def bind_inputs(
    definition: _QuantumFunction,
    supplied: Mapping[str, float | Sequence[float]] | None,
) -> dict[str, ClassicalValue]:
    """Validate names and snapshot bounded numeric arrays before program capture."""
    parameters = classical_parameters(definition)
    values = {} if supplied is None else supplied
    if not isinstance(values, Mapping) or set(values) != set(parameters):
        _error(
            definition.source,
            definition.source.node,
            "program.binding",
            f"classical inputs must match {sorted(parameters)}",
        )
    bound: dict[str, ClassicalValue] = {}
    elements = 0
    for name, parameter in parameters.items():
        value = values[name]
        check = _checked_integer if parameter.integer else _checked_real
        if parameter.array:
            if type(value) not in (list, tuple) or not isinstance(value, (list, tuple)):
                _error(
                    definition.source,
                    definition.source.node,
                    "program.binding",
                    f"classical array {name!r} requires a built-in numeric list or tuple",
                )
            elements += len(value)
            if not 1 <= len(value) <= 4096 or elements > 4096:
                _error(
                    definition.source,
                    definition.source.node,
                    "budget",
                    "classical arrays require 1..4096 elements and at most 4096 in total",
                )
            bound[name] = tuple(
                check(definition, definition.source.node, item) for item in value
            )
        else:
            elements += 1
            if elements > 4096:
                _error(
                    definition.source,
                    definition.source.node,
                    "budget",
                    "too many bindings",
                )
            bound[name] = check(definition, definition.source.node, value)
    return bound


def classical_number(
    owner: _Body, node: ast.expr, depth: int = 0
) -> int | float | None:
    """Interpret finite scalar arithmetic, never call host Python or read a qubit."""
    if depth >= 32:
        _error(owner.source, node, "budget", "classical expression depth exceeds 32")
    value: object = None
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name) and node.id in owner.classical:
        value = owner.classical[node.id]
    elif isinstance(node, ast.Name) and node.id in owner.loop_variables:
        value = owner.loop_variables[node.id]
    elif isinstance(node, (ast.Name, ast.Attribute)):
        visible = owner.aliases if owner.aliases is not None else owner.ports
        base: ast.expr = node
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and (
            base.id in visible
            or base.id in owner.classical
            or base.id in owner.loop_variables
        ):
            return None
        value = _resolve(owner.definition, node)
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name) or node.value.id not in owner.classical:
            return None
        array = owner.classical[node.value.id]
        if not isinstance(array, tuple) or isinstance(node.slice, ast.Slice):
            _error(
                owner.source,
                node,
                "program.binding",
                "index a one-dimensional numeric array",
            )
        from .frontend_views import static_integer

        index = static_integer(owner, node.slice, depth + 1)
        if not 0 <= index < len(array):
            _error(
                owner.source,
                node,
                "program.binding",
                "classical array index is out of bounds",
            )
        value = array[index]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = classical_number(owner, node.operand, depth + 1)
        if operand is None:
            return None
        value = -operand if isinstance(node.op, ast.USub) else operand
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = classical_number(owner, node.left, depth + 1)
        right = classical_number(owner, node.right, depth + 1)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        value = (
            left + right
            if isinstance(node.op, ast.Add)
            else (
                left - right
                if isinstance(node.op, ast.Sub)
                else left * right if isinstance(node.op, ast.Mult) else left / right
            )
        )
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(converted) else None


def real_number(owner: _Body, node: ast.expr, depth: int = 0) -> float | None:
    """Convert an admitted classical scalar only at a real-valued use site."""
    value = classical_number(owner, node, depth)
    return None if value is None else float(value)


def bind_helper(
    owner: _Body, node: ast.Call, definition: _QuantumFunction
) -> tuple[dict[str, ClassicalValue], tuple[ast.expr, ...]]:
    """Separate immutable classical actuals from checked quantum borrows."""
    formals = (
        *definition.source.node.args.posonlyargs,
        *definition.source.node.args.args,
    )
    if len(formals) != len(node.args):
        _error(
            owner.source, node, "program.signature", "qfunc argument count disagrees"
        )
    parameters = classical_parameters(definition)
    bindings: dict[str, ClassicalValue] = {}
    quantum: list[ast.expr] = []
    elements = 0
    for formal, actual in zip(formals, node.args, strict=True):
        if formal.arg not in parameters:
            quantum.append(actual)
            continue
        parameter = parameters[formal.arg]
        check = _checked_integer if parameter.integer else _checked_real
        if parameter.array:
            if isinstance(actual, ast.Name) and isinstance(
                owner.classical.get(actual.id), tuple
            ):
                value = owner.classical[actual.id]
                assert isinstance(value, tuple)
            elif isinstance(actual, (ast.List, ast.Tuple)):
                if not 1 <= len(actual.elts) <= 4096:
                    _error(
                        owner.source,
                        actual,
                        "budget",
                        "numeric arrays require 1..4096 elements",
                    )
                items = tuple(classical_number(owner, item) for item in actual.elts)
                if any(item is None for item in items):
                    _error(
                        owner.source,
                        actual,
                        "program.binding",
                        "array elements require finite static numbers",
                    )
                value = tuple(item for item in items if item is not None)
            else:
                _error(
                    owner.source,
                    actual,
                    "program.binding",
                    "pass a bound numeric array or literal array",
                )
            elements += len(value)
            bindings[formal.arg] = tuple(
                check(owner.definition, actual, item) for item in value
            )
        else:
            number = classical_number(owner, actual)
            bindings[formal.arg] = check(owner.definition, actual, number)
            elements += 1
        if elements > 4096:
            _error(
                owner.source,
                actual,
                "budget",
                "helper classical values exceed 4096 elements",
            )
    return bindings, tuple(quantum)
