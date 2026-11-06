"""Lexical register-view bindings without allocating or reversing quantum data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from .syntax import MAX_SCOPE_DEPTH, Expression, Parameter, Statement, Token, reject
from .unitary_effects import PURE_EFFECTS, emit_inverse

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
    from .adapter import _Adapt
    from .syntax import Parser


def is_binding_scope(statement: Statement) -> bool:
    return any(child.token.text == "->" for child in statement.within or ())


def parse_within(parser: Parser, allow_scopes: bool, scope_depth: int) -> Statement:
    token = parser.take("within")
    if not allow_scopes:
        reject(token, "scope", "nested scopes belong in apply, not preparation")
    if scope_depth >= MAX_SCOPE_DEPTH:
        reject(token, "budget.depth", f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}")
    preparation = parser.block(
        allow_scopes=False, scope_depth=scope_depth + 1, binding_preparation=True
    )
    parser.take("apply")
    has_bindings = any(s.token.text == "->" for s in preparation)
    actions = parser.block(scope_depth=scope_depth + 1, binding_actions=has_bindings)
    return Statement(token, (), within=preparation, apply=actions)


def lower_binding_scope(owner: _Adapt, statement: Statement, index: str) -> bool:
    if not is_binding_scope(statement):
        return False
    views = dict(owner.views)
    widths = dict(owner.widths)
    suspended = set(owner.suspended)
    start = len(owner.operations)
    reversible = []
    try:
        for ordinal, child in enumerate(statement.within or ()):
            if child.token.text == "->":
                owner.bind_registers(child.token, *child.arguments)
            else:
                before = len(owner.operations)
                owner.statement(child, f"{index}.binding_prepare{ordinal}")
                reversible.extend(owner.operations[before:])
        prepared_registers = dict(owner.registers)
        prepared_views = dict(owner.views)
        prepared_suspended = set(owner.suspended)
        for ordinal, child in enumerate(statement.apply):
            owner.statement(child, f"{index}.binding{ordinal}")
        if (
            any(op.name not in PURE_EFFECTS for op in owner.operations[start:])
            or owner.views != prepared_views
            or owner.suspended != prepared_suspended
            or owner.registers != prepared_registers
        ):
            reject(
                statement.token,
                "binding.scope",
                "binding apply admits only static gates; allocation and ownership changes refuse",
            )
        emit_inverse(owner, statement.token, reversible)
    finally:
        owner.views = views
        owner.widths = widths
        owner.suspended = suspended
    return True


def record_view(
    owner: _Adapt, parameter: Parameter, argument: Expression
) -> tuple[tuple[Value, int], ...]:
    root = argument.children[0] if argument.token.text == ":" else argument
    nested = owner.record_field(root)
    actual = (
        nested[0]
        if nested is not None
        else (None if root.children else owner.parameters.get(root.token.text))
    )
    if (
        actual is not None
        and actual.record is not None
        and actual.record != parameter.record
    ):
        reject(
            argument.token,
            "signature",
            "record helper requires the matching record type",
        )
    if actual is None or (
        actual.record != parameter.record
        and not (actual.record is None and actual.kind == "qbit[]")
    ):
        reject(
            argument.token,
            "signature",
            "record helper requires its nominal record or an exact untyped array view",
        )
    view = owner.array_view(argument)
    if len(view) != parameter.width:
        reject(
            argument.token, "width", "record view must match its exact declared width"
        )
    return view


def publish_bound_array(
    owner: _Adapt, token: Token, view: tuple[tuple[Value, int], ...]
) -> None:
    value = view[0][0]
    declaration = next(
        (
            op
            for op in owner.operations
            if op.name == "qreg.hidden" and op.results == (value,)
        ),
        None,
    )
    if (
        owner.scope != "main"
        or owner.parameters[token.text].kind != "qbit[]"
        or declaration is None
        or view != tuple((value, index) for index in range(register_width(value)))
    ):
        reject(
            token,
            "binding",
            "fresh output binding requires a whole local array in its original order",
        )
    # The output declaration introduces this value before its existing allocation.
    owner.operations.remove(declaration)
    owner.registers[token.text] = value


def publish_partition(
    owner: _Adapt,
    token: Token,
    source: Expression,
    destination: Expression,
) -> bool:
    destinations = (
        destination.children if destination.token.text == "{" else (destination,)
    )
    names = tuple(owner.atom(item).text for item in destinations)
    if any(
        name not in owner.parameters
        or owner.parameters[name].role != "output"
        or name in owner.registers
        or name in owner.views
        for name in names
    ):
        return False
    sources = source.children if source.token.text == "{" else (source,)
    source_names = tuple(owner.atom(item).text for item in sources)
    if len(set((*names, *source_names))) != len(names) + len(source_names):
        reject(token, "binding", "publication names must be distinct")
    view = tuple(location for item in sources for location in owner.array_view(item))
    owners = tuple(dict.fromkeys(value for value, _ in view))
    if (
        len(names) == 1
        and len(owners) == 1
        and owner.parameters[names[0]].kind == "qbit[]"
    ):
        # Keep the existing complete, ordered single-owner array contract.
        return False
    if any(
        not any(
            op.name == "qreg.hidden" and op.results == (value,)
            for op in owner.operations
        )
        for value in owners
    ):
        reject(token, "binding", "publication requires complete owned hidden storage")
    widths = tuple(owner.widths[name] for name in names)
    if widths.count(None) > 1:
        reject(token, "binding", "publication may infer at most one output width")
    inferred = len(view) - sum(width for width in widths if width is not None)
    if (None in widths and inferred <= 0) or (None not in widths and inferred != 0):
        reject(token, "binding", "publication widths must cover the source exactly")
    outputs = tuple(
        owner.register(
            item.token,
            inferred if width is None else width,
            allow_qubit_array=True,
            allow_fractional_output=True,
        )
        for item, width in zip(destinations, widths, strict=True)
    )
    for name, value in zip(names, outputs, strict=True):
        if owner.parameters[name].kind in {"qphase", "qphase_signed"}:
            owner.fractional_bits[name] = register_width(value)
    owner_indices = {value: index for index, value in enumerate(owners)}
    locations = tuple((owner_indices[value], bit) for value, bit in view)
    owner.effect_op(
        "qreg.publish",
        token,
        (*outputs, *owners),
        destinations=len(outputs),
        locations=locations,
    )
    owner.suspended.update(source_names)
    return True
