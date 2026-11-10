"""Capture bounded raw-word modular updates using checked indexed X effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_classical import bind_helper
from .frontend_numeric import has_numeric, reject_numeric_reads
from .frontend_ports import program_ports
from .frontend_python import _error, _QuantumFunction
from .frontend_views import locations, static_integer
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def _emit(
    owner: _Body,
    node: ast.expr | ast.stmt,
    target: Value,
    amount: int,
    control: tuple[Value, int] | None = None,
) -> None:
    if abs(amount) > 256:
        _error(owner.source, node, "program.increment", "offset must fit -256..256")
    assert isinstance(node, (ast.expr, ast.stmt))
    width = register_width(target)
    offset = amount % (1 << width)
    if offset > (1 << width) // 2:
        offset -= 1 << width
    # A zero update still witnesses initialized writable storage in the verifier.
    if offset == 0:
        if control is None:
            owner.emit("qreg.gate", node, (target,), gate="i", index=0)
        else:
            for _ in range(2):
                owner.emit(
                    "qreg.control_x",
                    node,
                    (control[0], target),
                    indices=(control[1], 0),
                )
    for _ in range(abs(offset)):
        for bit in (range(width) if offset < 0 else reversed(range(width))):
            selected = (() if control is None else (control,)) + tuple(
                (target, lower) for lower in range(bit)
            )
            if selected:
                selected += ((target, bit),)
                owner.emit(
                    "qreg.control_x",
                    node,
                    tuple(value for value, _ in selected),
                    indices=tuple(index for _, index in selected),
                )
            else:
                owner.emit("qreg.gate", node, (target,), gate="x", index=bit)


def _target(owner: _Body, node: ast.expr) -> Value:
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    if (
        not isinstance(node, ast.Name)
        or node.id not in ports
        or ports[node.id].kind != "qnum"
        or has_numeric(ports[node.id])
        or ports[node.id].role not in {"output", "inout"}
    ):
        _error(
            owner.source, node, "program.increment", "use a writable raw unsigned QNum"
        )
    view = locations(owner, node)
    value = view[0][0]
    if view != tuple((value, bit) for bit in range(register_width(value))):
        _error(
            owner.source,
            node,
            "program.increment",
            "target must cover a complete register",
        )
    if owner.aliases is not None and any(
        actual == value and owner.ports[name].kind != "qnum"
        for name, actual in owner.registers.items()
    ):
        _error(
            owner.source, node, "program.increment", "use a writable raw unsigned QNum"
        )
    return value


def capture_increment(owner: _Body, node: ast.AugAssign) -> None:
    target = _target(owner, node.target)
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
    if ports[0].width is not None and ports[0].width != register_width(target):
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
    if statement.target.id != ports[0].name or target in condition_inputs:
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
