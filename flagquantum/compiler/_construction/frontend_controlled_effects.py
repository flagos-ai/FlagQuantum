"""Capture pure borrowed callbacks before shared controlled-effect expansion."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_callback_effects import emit_callback_effects
from .frontend_classical import bind_helper
from .frontend_evolution import flush_evolutions
from .frontend_ports import program_ports
from .frontend_python import _error, _QuantumFunction
from .frontend_views import locations, packed_locations
from .register_profile import _verify_borrowed_effects, register_width

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
    # A raw scalar control borrows one location; computed predicates retain
    # whole dependency registers until their compute/uncompute region closes.
    raw_scalar = inputs == (predicate,)
    protected = (
        {(predicate, index)}
        if raw_scalar
        else {(value, bit) for value in inputs for bit in range(register_width(value))}
    )
    if any(
        any(location in protected for location in view)
        and (not formal or formal[position].role != "const")
        for position, view in enumerate(views)
    ):
        _error(
            owner.source,
            callback,
            "program.control",
            "condition inputs and writable callback actual storage must be disjoint",
        )
    flush_evolutions(owner)
    storage = dict(owner.registers), dict(owner.ports), dict(owner.widths)
    start, incoming = len(owner.operations), owner.effect
    owner.region_depth += 1
    owner.owner.expanded_callbacks += 1
    try:
        owner.call(callback)
        flush_evolutions(owner)
    finally:
        owner.owner.expanded_callbacks -= 1
        owner.region_depth -= 1
    effects = tuple(owner.operations[start:])
    del owner.operations[start:]
    owner.effect = incoming
    if not effects or storage != (owner.registers, owner.ports, owner.widths):
        _error(
            owner.source,
            callback,
            "program.control",
            "controlled callbacks require pure effects on existing storage without ownership changes or escaping scratch",
        )
    mutable_values = {
        value
        for position, view in enumerate(views)
        if not formal or formal[position].role != "const"
        for value, _ in view
    }
    readonly = tuple(
        value
        for value in inputs
        if not (raw_scalar and value == predicate and value in mutable_values)
    )
    # The exact view admission above excludes selector writes. A disjoint
    # writable view may share its backing Value with that scalar selector;
    # model the mixed backing register as initialized mutable storage in the
    # existing lifecycle envelope. Normal indexed control verification still
    # applies after emission, including all expanded helper effects.
    writable = tuple(
        dict.fromkeys(
            value for view in views for value, _ in view if value not in readonly
        )
    )
    # Retain authoritative Const and lifecycle checks after exact view admission.
    # Computed dependencies stay read-only; a raw selector may share its backing
    # register with independently writable bit locations.
    _verify_borrowed_effects(
        effects, owner.owner.functions, tuple(dict.fromkeys(readonly)), writable
    )
    try:
        emit_callback_effects(owner, callback, effects, predicate, index)
    except ValueError as error:
        _error(owner.source, callback, "program.control", str(error))
