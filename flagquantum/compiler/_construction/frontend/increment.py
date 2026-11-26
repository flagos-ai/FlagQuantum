"""Capture bounded raw-word modular updates using checked indexed X effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from flagquantum.compiler.frontends._structured.model import Value

from .classical import bind_helper
from .numeric import has_numeric, reject_numeric_reads
from .ports import program_ports
from .python import _error, _QuantumFunction
from .views import locations, packed_locations, static_integer

if TYPE_CHECKING:
    from .program import _Body


def _emit(
    owner: _Body,
    node: ast.expr | ast.stmt,
    target: tuple[tuple[Value, int], ...],
    amount: int,
    control: tuple[Value, int] | None = None,
) -> None:
    if abs(amount) > 256:
        _error(owner.source, node, "program.increment", "offset must fit -256..256")
    assert isinstance(node, (ast.expr, ast.stmt))
    width = len(target)
    offset = amount % (1 << width)
    if offset > (1 << width) // 2:
        offset -= 1 << width
    # A zero update still witnesses initialized writable storage in the verifier.
    if offset == 0:
        if control is None:
            owner.emit("qreg.gate", node, (target[0][0],), gate="i", index=target[0][1])
        else:
            for _ in range(2):
                owner.emit(
                    "qreg.control_x",
                    node,
                    (control[0], target[0][0]),
                    indices=(control[1], target[0][1]),
                )
    for _ in range(abs(offset)):
        bits = range(width) if offset < 0 else reversed(range(width))
        for bit in bits:
            selected = (() if control is None else (control,)) + tuple(
                target[lower] for lower in range(bit)
            )
            if selected:
                selected += (target[bit],)
                owner.emit(
                    "qreg.control_x",
                    node,
                    tuple(value for value, _ in selected),
                    indices=tuple(index for _, index in selected),
                )
            else:
                owner.emit(
                    "qreg.gate", node, (target[bit][0],), gate="x", index=target[bit][1]
                )


def _target(
    owner: _Body, node: ast.expr, *, writable: bool = True
) -> tuple[tuple[Value, int], ...]:
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    message = (
        "use a writable raw unsigned QNum" if writable else "use a raw unsigned QNum"
    )
    explicit = isinstance(node, (ast.List, ast.Tuple))
    base = node
    while isinstance(base, ast.Subscript):
        base = base.value
    if not explicit and (
        not isinstance(base, ast.Name)
        or base.id not in ports
        or ports[base.id].kind != "qnum"
    ):
        _error(owner.source, node, "program.increment", message)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in ports:
            port = ports[child.id]
            if has_numeric(port) or (writable and port.role not in {"output", "inout"}):
                _error(
                    owner.source,
                    child,
                    "program.increment",
                    message,
                )
    view = packed_locations(owner, node) if explicit else locations(owner, node)
    if not 1 <= len(view) <= 256 or len(set(view)) != len(view):
        _error(
            owner.source,
            node,
            "program.increment",
            "numeric views require 1..256 distinct locations",
        )
    for value, _ in view:
        actual = next(
            port
            for name, port in owner.ports.items()
            if owner.registers.get(name) == value
        )
        if has_numeric(actual) or (writable and actual.role not in {"output", "inout"}):
            _error(
                owner.source,
                node,
                "program.increment",
                message,
            )
    return view


def capture_increment(owner: _Body, node: ast.AugAssign) -> None:
    target = _target(owner, node.target)
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    quantum_source = any(
        isinstance(child, ast.Name) and child.id in ports
        for child in ast.walk(node.value)
    )
    if quantum_source:
        source = _target(owner, node.value, writable=False)
        if set(source) & set(target):
            _error(
                owner.source,
                node,
                "program.increment",
                "quantum addend and accumulator must be physically disjoint",
            )
        # Logical view element zero is the low bit. Each source bit controls a
        # unit increment of the suffix starting at its weight, modulo target width.
        # Higher addend bits vanish modulo that width; no carry storage is needed.
        direction = -1 if isinstance(node.op, ast.Sub) else 1
        for bit, selector in enumerate(source[: len(target)]):
            _emit(owner, node, target[bit:], direction, selector)
        return
    amount = static_integer(owner, node.value)
    if isinstance(node.op, ast.Sub):
        amount = -amount
    _emit(owner, node, target, amount)


def is_increment_helper(function: object) -> bool:
    if not isinstance(function, _QuantumFunction) or not function.source.program:
        return False
    statements = function.source.node.body
    return (
        len(statements) == 1
        and isinstance(statements[0], ast.AugAssign)
        and isinstance(statements[0].op, (ast.Add, ast.Sub))
        and isinstance(statements[0].target, ast.Name)
    )


def capture_increment_call(
    owner: _Body,
    call: ast.Call,
    function: _QuantumFunction,
    predicate: tuple[Value, int],
    condition_inputs: tuple[Value, ...],
) -> None:
    owner.owner.inline_calls += 1
    if owner.owner.inline_calls > 256:
        _error(
            owner.source, call, "budget", "mutable helper expansion exceeds 256 calls"
        )
    if id(function) not in owner.owner.seen:
        owner.owner.seen.add(id(function))
        owner.owner.nodes += sum(1 for _ in ast.walk(function.source.node))
        if owner.owner.nodes > 4096:
            _error(
                owner.source,
                call,
                "budget",
                "program definitions exceed 4096 AST nodes",
            )
    ports = program_ports(function)
    if (
        len(ports) != 1
        or ports[0].role != "inout"
        or ports[0].kind != "qnum"
        or has_numeric(ports[0])
    ):
        _error(
            owner.source,
            call,
            "program.increment",
            "controlled update requires one raw inout QNum",
        )
    classical, actuals = bind_helper(owner, call, function)
    reject_numeric_reads(owner, actuals[0], "controlled increment")
    target = _target(owner, actuals[0])
    if ports[0].width is not None and ports[0].width != len(target):
        _error(
            owner.source,
            call,
            "program.width",
            "helper formal and actual widths disagree",
        )
    statement = function.source.node.body[0]
    assert isinstance(statement, ast.AugAssign) and isinstance(
        statement.target, ast.Name
    )
    # Computed predicates borrow complete source Values; a direct selector
    # borrows one physical bit, including through nested helper aliases.
    direct_scalar = condition_inputs == (predicate[0],)
    if statement.target.id != ports[0].name or any(
        (value, bit) == predicate if direct_scalar else value in condition_inputs
        for value, bit in target
    ):
        _error(
            owner.source,
            call,
            "program.increment",
            "condition and target must be disjoint",
        )
    saved = owner.definition, owner.source, owner.classical, owner.loop_variables
    owner.definition, owner.source, owner.classical, owner.loop_variables = (
        function,
        function.source,
        classical,
        {},
    )
    try:
        amount = static_integer(owner, statement.value)
        if isinstance(statement.op, ast.Sub):
            amount = -amount
        _emit(owner, statement, target, amount, predicate)
    finally:
        owner.definition, owner.source, owner.classical, owner.loop_variables = saved
