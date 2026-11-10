"""Capture pure borrowed callbacks before shared controlled-effect expansion."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .._hybrid.unitary_effects import PURE_EFFECTS
from .frontend_classical import bind_helper
from .frontend_evolution import flush_evolutions
from .frontend_python import _error, _QuantumFunction
from .frontend_views import locations
from .qpe_control import emit_controlled_effect

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_controlled_effects(
    owner: _Body,
    callback: ast.Call,
    predicate: Value,
    index: int,
    inputs: tuple[Value, ...],
) -> None:
    """Admit existing-storage pure effects while preserving branch global phase."""
    function = owner.resolve_callable(callback.func)
    if isinstance(function, _QuantumFunction):
        _, arguments = bind_helper(owner, callback, function)
    else:
        if len(callback.args) != 3:
            _error(
                owner.source,
                callback,
                "program.signature",
                "inplace preparation requires a vector, bound and target",
            )
        arguments = (callback.args[-1],)
    if any(
        value in inputs
        for argument in arguments
        for value, _ in locations(owner, argument)
    ):
        _error(
            owner.source,
            callback,
            "program.control",
            "condition inputs and callback actual storage must be disjoint",
        )
    flush_evolutions(owner)
    storage = dict(owner.registers), dict(owner.ports), dict(owner.widths)
    start, incoming = len(owner.operations), owner.effect
    owner.region_depth += 1
    try:
        owner.call(callback)
        flush_evolutions(owner)
    finally:
        owner.region_depth -= 1
    effects = tuple(owner.operations[start:])
    del owner.operations[start:]
    owner.effect = incoming
    if (
        not effects
        or storage != (owner.registers, owner.ports, owner.widths)
        or any(operation.name not in PURE_EFFECTS for operation in effects)
    ):
        _error(
            owner.source,
            callback,
            "program.control",
            "controlled callbacks require pure effects on existing storage without ownership changes or escaping scratch",
        )
    if any(set(operation.operands[1:]).intersection(inputs) for operation in effects):
        _error(
            owner.source,
            callback,
            "program.control",
            "condition inputs and callback storage must be disjoint",
        )

    def emit(
        name: str, values: tuple[Value, ...], attributes: dict[str, object]
    ) -> None:
        owner.emit(name, callback, values, **attributes)

    try:
        for operation in effects:
            emit_controlled_effect(predicate, index, operation, emit)
    except ValueError as error:
        _error(owner.source, callback, "program.control", str(error))
