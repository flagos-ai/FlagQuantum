"""Bounded lexical callback arrays composed from checked source effects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._hybrid.model import Operation
from .quantum_callbacks import MAX_CALLBACK_STATEMENTS, Lambda
from .syntax import Expression, Function, Statement, Token, reject
from .unitary_effects import PURE_EFFECTS

if TYPE_CHECKING:
    from .adapter import _Adapt

MAX_CALLBACK_ARRAY_LENGTH = 64


def capture_callbacks(
    owner: _Adapt, argument: Expression
) -> tuple[tuple[Operation, ...], ...]:
    if not argument.children and argument.token.text in owner.callback_arrays:
        return owner.callback_arrays[argument.token.text]
    if (
        argument.token.text != "[]"
        or not 1 <= len(argument.children) <= MAX_CALLBACK_ARRAY_LENGTH
    ):
        reject(
            argument.token,
            "callback.signature",
            "expected a literal array of 1..64 callbacks",
        )
    captured = []
    total_effects = 0
    for callback in argument.children:
        if not isinstance(callback, Lambda) or callback.parameters or not callback.body:
            reject(
                callback.token,
                "callback.signature",
                "callback array requires nonempty zero-argument lambdas",
            )
        location = callback.token.location
        syntax = Function(
            Token(f"$capture_{location.line}_{location.column}", location),
            tuple(owner.parameters.values()),
            callback.body,
        )
        child = type(owner)(
            syntax,
            owner.resolve,
            owner.definitions,
            owner.constants,
            owner.record_types,
        )
        child.classical_values = dict(owner.classical_values)
        child.classical_arrays = dict(owner.classical_arrays)
        child.classical_integers = dict(owner.classical_integers)
        child.integer_arrays = dict(owner.integer_arrays)
        child.loop_indices = dict(owner.loop_indices)
        child.widths = dict(owner.widths)
        child.registers = dict(owner.registers)
        child.views = dict(owner.views)
        child.suspended = set(owner.suspended)
        child.dropped = set(owner.dropped)
        child.signed = set(owner.signed)
        child.fractional_bits = dict(owner.fractional_bits)
        child.helper_stack = owner.helper_stack
        child.helper_expansions = owner.helper_expansions
        child.loop_iterations = owner.loop_iterations
        for ordinal, statement in enumerate(callback.body):
            child.statement(statement, str(ordinal))
        if (
            any(op.name not in PURE_EFFECTS for op in child.operations)
            or child.functions
            or child.registers != owner.registers
            or child.views != owner.views
            or child.suspended != owner.suspended
        ):
            reject(
                callback.token,
                "callback.escape",
                "lexical callback requires pure effects on existing storage",
            )
        total_effects += len(child.operations)
        if total_effects > MAX_CALLBACK_STATEMENTS:
            reject(
                callback.token,
                "budget.callback",
                "callback array exceeds 4096 captured effects",
            )
        owner.helper_expansions = child.helper_expansions
        owner.loop_iterations = child.loop_iterations
        captured.append(tuple(child.operations))
    return tuple(captured)


def indexed_callback(owner: _Adapt, statement: Statement) -> tuple[Operation, ...]:
    assert statement.callback_index is not None
    callbacks = owner.callback_arrays.get(statement.token.text)
    if callbacks is None:
        reject(
            statement.token,
            "callback.signature",
            "indexed calls require a bound callable array",
        )
    index = owner.static_integer(statement.callback_index)
    if not 0 <= index < len(callbacks):
        reject(
            statement.token, "callback.index", "callable array index is out of bounds"
        )
    return callbacks[index]


def lower_callback_control(owner: _Adapt, statement: Statement) -> bool:
    body = statement.control_body
    if body is None or len(body) != 1 or body[0].callback_index is None:
        return False
    condition = statement.arguments[0]
    if condition.token.text != "==" or statement.control_else is not None:
        reject(
            statement.token,
            "callback.control",
            "indexed callback control requires one equality and no else",
        )
    lower_selected_effects(
        owner, statement.token, condition, indexed_callback(owner, body[0])
    )
    return True


def lower_selected_effects(
    owner: _Adapt, token: Token, condition: Expression, effects: tuple[Operation, ...]
) -> None:
    selector, literal = condition.children
    controls = owner.array_view(selector)
    expected = owner.static_integer(literal)
    if len(controls) > 7 or not 0 <= expected < 1 << len(controls):
        reject(
            literal.token,
            "callback.control",
            "selector must be at most seven bits and equality must fit",
        )
    negative = tuple(
        location for bit, location in enumerate(controls) if not (expected >> bit) & 1
    )
    for value, index in negative:
        owner.effect_op("qreg.gate", token, (value,), gate="x", index=index)
    for operation in effects:
        operands = operation.operands[1:]
        attributes = operation.attributes
        if operation.name == "qreg.control_x" and "indices" in attributes:
            locations = tuple(zip(operands, attributes["indices"], strict=True))
            owner.emit_indexed_x(token, (*controls, *locations))
            continue
        if operation.name != "qreg.gate":
            reject(
                token,
                "callback.control",
                "controlled callback requires indexed X or a supported scalar gate",
            )
        value = operands[0]
        index = attributes["index"]
        if (value, index) in controls:
            reject(
                token, "alias", "controlled callback must be disjoint from its selector"
            )
        gate = attributes["gate"]
        if gate == "i":
            owner.effect_op("qreg.gate", token, (value,), gate="i", index=index)
        elif gate == "x":
            owner.emit_indexed_x(token, (*controls, (value, index)))
        elif gate in {"rx", "ry"}:
            angle = attributes["angle"] / 2
            if attributes.get("inverse"):
                angle = -angle
            locations = (*controls, (value, index))
            for direction in (1, -1):
                owner.effect_op(
                    "qreg.gate",
                    token,
                    (value,),
                    gate=gate,
                    index=index,
                    angle=direction * angle,
                )
                owner.effect_op(
                    "qreg.control_phase",
                    token,
                    tuple(register for register, _ in locations),
                    indices=tuple(bit for _, bit in locations),
                    angle=math.pi,
                )
        else:
            reject(
                token,
                "callback.control",
                "controlled callback scalar gate must be I, X, RX or RY",
            )
    for value, index in reversed(negative):
        owner.effect_op("qreg.gate", token, (value,), gate="x", index=index)
