"""Capture reversible native regions using the shared Compiler inverse rules."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from .._hybrid.model import Operation, Value, ValueId
from .._hybrid.unitary_effects import inverse_effects, unitary_regions
from .frontend_evolution import flush_evolutions
from .frontend_ports import program_ports
from .frontend_python import _error, _QuantumFunction

if TYPE_CHECKING:
    from .frontend_program import _Body


def _checked_effects(
    owner: _Body, node: ast.With, statements: Sequence[ast.stmt]
) -> tuple[Operation, ...]:
    if owner.region_depth >= 32:
        _error(owner.source, node, "budget", "unitary region nesting exceeds 32")
    storage = dict(owner.registers), dict(owner.ports), dict(owner.widths)
    start = len(owner.operations)
    owner.region_depth += 1
    try:
        owner.statements(statements, nested=True)
        flush_evolutions(owner)
    finally:
        owner.region_depth -= 1
    effects = tuple(owner.operations[start:])
    try:
        unitary_regions(effects)
    except ValueError as error:
        _error(owner.source, node, "program.region", str(error))
    if storage != (owner.registers, owner.ports, owner.widths) or any(
        isinstance(child, ast.Return)
        for statement in statements
        for child in ast.walk(statement)
    ):
        _error(
            owner.source,
            node,
            "program.region",
            "unitary regions require pure effects on existing initialized storage; "
            "allocation, Output publication and ownership changes are not admitted",
        )
    return effects


def _emit_inverse(owner: _Body, node: ast.With, effects: Sequence[Operation]) -> None:
    try:
        backward = inverse_effects(effects)
    except ValueError as error:
        _error(owner.source, node, "program.adjoint", str(error))
    local_bindings: dict[Value, Value] = {}
    for op in backward:
        if op.name == "qreg.local":
            original = op.results[0]
            fresh = Value(ValueId(owner.name, owner.counter), original.type)
            owner.counter += 1
            local_bindings[original] = fresh
            owner.operations.append(
                replace(
                    op,
                    results=(fresh,),
                    attributes={"name": f"adjoint.{fresh.id.index}"},
                )
            )
        else:
            owner.emit(
                op.name,
                node,
                tuple(local_bindings.get(v, v) for v in op.operands[1:]),
                **op.attributes,
            )


def capture_unitary_within(owner: _Body, prepare: ast.With, action: ast.With) -> None:
    """Emit U V U-dagger without changing register bindings or ownership."""
    flush_evolutions(owner)
    forward = _checked_effects(owner, prepare, prepare.body)
    _checked_effects(owner, action, action.body)
    _emit_inverse(owner, prepare, forward)


def capture_adjoint(owner: _Body, block: ast.With) -> None:
    """Keep Const-only calls and invert expanded pure gates on mutable storage."""
    flush_evolutions(owner)
    calls = [
        statement.value
        for statement in block.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    # Const functions can contain checked arithmetic locals with their own cleanup.
    # Keep the existing callee-level inverse path for those atomic calls.
    if len(calls) == len(block.body) and all(
        isinstance(function := owner.resolve_callable(call.func), _QuantumFunction)
        and function.source.program
        and all(port.role == "const" for port in program_ports(function))
        for call in calls
    ):
        for call in reversed(calls):
            owner.call(call, inverse=True)
        return
    for call in calls:
        function = owner.resolve_callable(call.func)
        if isinstance(function, _QuantumFunction) and any(
            port.role == "output" for port in program_ports(function)
        ):
            _error(
                owner.source,
                call,
                "program.adjoint",
                "adjoint requires a static Const-only or mutable unitary qfunc; "
                "Output publication is not admitted",
            )
    start, effect = len(owner.operations), owner.effect
    forward = _checked_effects(owner, block, block.body)
    del owner.operations[start:]
    owner.effect = effect
    _emit_inverse(owner, block, forward)
