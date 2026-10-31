"""Verify a coherent-integer profile over the existing structured SSA records.

Values describe expressions, not separately allocated quantum registers. Repeated
reads do not clone a state. Physical resources are checked separately at binding.
This module neither emits arithmetic gates nor executes a program.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import NoReturn

from .._hybrid.model import INDEX, HybridProgram, IRType, Operation, ValueId
from .._hybrid.verifier import Diagnostic, HybridVerificationError
from .ranges import unsigned_width

_ATTRIBUTES = MappingProxyType(
    {
        "arith.constant": frozenset({"value"}),
        "qint.add": frozenset(),
        "qint.multiply": frozenset(),
        "qint.equal": frozenset(),
        "qint.less": frozenset(),
        "qint.call": frozenset({"callee"}),
        "qint.return": frozenset(),
    }
)


def uint_type(width: int) -> IRType:
    """Describe a basis-encoded unsigned quantum integer, distinct from INDEX."""
    _positive_integer(width, "width")
    return IRType("quantum.uint", (width,))


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _fail(code: str, message: str, operation: Operation | None = None) -> NoReturn:
    if operation is not None and operation.location is not None:
        loc = operation.location
        message = f"{loc.filename}:{loc.line}:{loc.column}: {message}"
    raise HybridVerificationError((Diagnostic(f"construction.{code}", message),))


def _width(value_type: IRType, max_width: int) -> int:
    if (
        value_type.kind != "quantum.uint"
        or value_type.linear
        or len(value_type.parameters) != 1
    ):
        _fail("type", "expected a non-consuming quantum.uint expression value")
    width = value_type.parameters[0]
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        _fail("type", "quantum.uint requires a positive integer width")
    if width > max_width:
        _fail("budget.width", f"width {width} exceeds safety limit {max_width}")
    return width


def verify_construction(
    functions: Sequence[HybridProgram],
    *,
    entry: str,
    max_width: int = 256,
    max_nodes: int = 4096,
    max_call_depth: int = 16,
) -> HybridProgram:
    """Check closed, acyclic expression functions without changing hybrid SSA.

    Each function borrows its quantum inputs and returns one integer expression.
    The eventual circuit computes that expression into a disjoint XOR target
    with zero-initialized workspace. This verifier does not prove a lowering.
    """
    for name, limit in (
        ("max_width", max_width),
        ("max_nodes", max_nodes),
        ("max_call_depth", max_call_depth),
    ):
        _positive_integer(limit, name)
    symbols: dict[str, HybridProgram] = {}
    node_count = 0
    for function in functions:
        if function.name in symbols:
            _fail("function.duplicate", f"duplicate function {function.name!r}")
        if len(function.body.blocks) != 1:
            _fail("function.blocks", "construction requires one block per function")
        block = function.body.blocks[0]
        node_count += 1 + len(block.arguments) + len(block.operations)
        if node_count > max_nodes:
            _fail("budget.nodes", f"definition nodes exceed {max_nodes}")
        symbols[function.name] = function
    if entry not in symbols:
        _fail("function.entry", f"unknown entry {entry!r}")

    # Cache return type, upper bound, expanded nodes and maximum call-stack depth.
    checked: dict[str, tuple[IRType, int, int, int]] = {}
    visiting: set[str] = set()

    def check(name: str) -> tuple[IRType, int, int, int]:
        if name in visiting:
            _fail("function.recursion", f"recursive call involving {name!r}")
        if name in checked:
            return checked[name]
        if len(visiting) >= max_call_depth:
            _fail("budget.depth", f"call depth exceeds {max_call_depth}")
        visiting.add(name)
        block = symbols[name].body.blocks[0]
        environment: dict[ValueId, tuple[IRType, int]] = {}
        for argument in block.arguments:
            width = _width(argument.type, max_width)
            if argument.id in environment:
                _fail("ssa.duplicate", f"duplicate argument {argument.id}")
            environment[argument.id] = (argument.type, (1 << width) - 1)
        if not block.operations or block.operations[-1].name != "qint.return":
            _fail("function.return", f"{name!r} requires a final qint.return")
        expanded, depth = len(block.operations), 1
        return_type: IRType | None = None
        return_bound = 0
        for index, operation in enumerate(block.operations):
            expected_attributes = _ATTRIBUTES.get(operation.name)
            if expected_attributes is None:
                _fail(
                    "operation.unsupported",
                    f"unsupported operation {operation.name!r}",
                    operation,
                )
            actual_attributes = set(operation.attributes)
            if actual_attributes != expected_attributes:
                _fail(
                    "operation.attributes",
                    f"{operation.name!r}: missing attributes "
                    f"{sorted(expected_attributes - actual_attributes)}, "
                    f"unsupported attributes {sorted(actual_attributes - expected_attributes)}",
                    operation,
                )
            if operation.regions:
                _fail("operation.regions", "nested regions are unsupported", operation)
            operands = []
            for operand in operation.operands:
                known = environment.get(operand.id)
                if known is None or known[0] != operand.type:
                    _fail("ssa.use", f"undefined or mistyped {operand.id}", operation)
                operands.append(known)
            if operation.name == "qint.return":
                if (
                    index != len(block.operations) - 1
                    or len(operands) != 1
                    or operation.results
                ):
                    _fail(
                        "function.return", "return requires one final value", operation
                    )
                _width(operands[0][0], max_width)
                return_type, return_bound = operands[0]
                continue
            if len(operation.results) != 1:
                _fail("ssa.result", "operation requires one result", operation)
            result = operation.results[0]
            if result.id in environment:
                _fail("ssa.duplicate", f"duplicate definition {result.id}", operation)
            if operation.name == "arith.constant":
                value = operation.attributes.get("value")
                if (
                    operands
                    or result.type != INDEX
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    _fail(
                        "constant",
                        "constant requires a nonnegative Python int and INDEX",
                        operation,
                    )
                if value.bit_length() > max_width:
                    _fail(
                        "budget.width",
                        "classical constant exceeds width budget",
                        operation,
                    )
                environment[result.id] = (INDEX, value)
                continue
            width = _width(result.type, max_width)
            if operation.name in {"qint.add", "qint.multiply"}:
                if len(operands) != 2 or all(item[0] == INDEX for item in operands):
                    _fail(
                        "arithmetic.signature",
                        "binary arithmetic requires a quantum operand",
                        operation,
                    )
                left, right = (item[1] for item in operands)
                bound = left + right if operation.name == "qint.add" else left * right
            elif operation.name in {"qint.equal", "qint.less"}:
                if len(operands) != 2 or all(item[0] == INDEX for item in operands):
                    _fail(
                        "comparison.signature",
                        "comparison requires two operands including a quantum value",
                        operation,
                    )
                if width != 1:
                    _fail(
                        "comparison.width",
                        "comparison result must be exactly one bit",
                        operation,
                    )
                bound = 1
            elif operation.name == "qint.call":
                callee = operation.attributes.get("callee")
                if not isinstance(callee, str) or callee not in symbols:
                    _fail(
                        "function.callee", "call requires a known function", operation
                    )
                expected = symbols[callee].body.blocks[0].arguments
                if tuple(item[0] for item in operands) != tuple(
                    arg.type for arg in expected
                ):
                    _fail(
                        "function.signature",
                        f"argument types disagree with {callee!r}",
                        operation,
                    )
                callee_type, bound, callee_nodes, callee_depth = check(callee)
                if result.type != callee_type:
                    _fail(
                        "function.signature",
                        "call result type disagrees with callee",
                        operation,
                    )
                expanded += callee_nodes
                depth = max(depth, 1 + callee_depth)
            else:
                _fail(
                    "operation.unsupported",
                    f"unsupported operation {operation.name!r}",
                    operation,
                )
            if unsigned_width(bound) > width:
                _fail(
                    "width.overflow",
                    f"result width {width} cannot hold upper bound {bound}",
                    operation,
                )
            if expanded > max_nodes or depth > max_call_depth:
                _fail(
                    "budget.expansion",
                    "expanded call graph exceeds construction budget",
                    operation,
                )
            environment[result.id] = (result.type, bound)
        visiting.remove(name)
        if return_type is None:  # Every accepted block ends with a checked return.
            _fail("function.return", f"missing result in {name!r}")
        checked[name] = (return_type, return_bound, expanded, depth)
        return checked[name]

    for name in symbols:
        check(name)
    return symbols[entry]
