"""Capture pure borrowed callbacks before shared controlled-effect expansion."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_callback_effects import emit_callback_effects
from .frontend_classical import bind_helper
from .frontend_ports import program_ports
from .frontend_python import _error, _QuantumFunction
from .frontend_views import locations, packed_locations
from .register_profile import _verify_borrowed_effects

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
        formal = program_ports(function)
    else:
        if len(callback.args) != 3:
            _error(
                owner.source,
                callback,
                "program.signature",
                "inplace preparation requires a vector, bound and target",
            )
        arguments = (callback.args[-1],)
        formal = ()
    views = tuple(
        (
            packed_locations(owner, argument)
            if not formal or formal[position].kind == "qnum"
            else locations(owner, argument)
        )
        for position, argument in enumerate(arguments)
    )
    if any(
        any(value in inputs for value, _ in view)
        and (not formal or formal[position].role != "const")
        for position, view in enumerate(views)
    ):
        _error(
            owner.source,
            callback,
            "program.control",
            "condition inputs and writable callback actual storage must be disjoint",
        )
    storage = dict(owner.registers), dict(owner.ports), dict(owner.widths)
    effects = owner.capture_effects(
        lambda: owner.call(callback), expanded_callback=True
    )
    if not effects or storage != (owner.registers, owner.ports, owner.widths):
        _error(
            owner.source,
            callback,
            "program.control",
            "controlled callbacks require pure effects on existing storage without ownership changes or escaping scratch",
        )
    writable = tuple(
        dict.fromkeys(
            value for view in views for value, _ in view if value not in inputs
        )
    )
    # The shared verifier proves that controls and their dependency registers
    # cannot be written, including through nested arithmetic and helper calls.
    _verify_borrowed_effects(
        effects, owner.owner.functions, tuple(dict.fromkeys(inputs)), writable
    )
    try:
        emit_callback_effects(owner, callback, effects, predicate, index)
    except ValueError as error:
        _error(owner.source, callback, "program.control", str(error))
