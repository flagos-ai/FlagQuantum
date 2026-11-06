"""Bounded lexical callback arrays composed from checked source effects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from ...compiler._hybrid.model import Operation
from .quantum_callbacks import MAX_CALLBACK_STATEMENTS, Lambda
from .syntax import Expression, Function, Statement, Token, reject
from .unitary_effects import PURE_EFFECTS

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
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
        effects = capture_effects(owner, callback.token, callback.body)
        total_effects += len(effects)
        if total_effects > MAX_CALLBACK_STATEMENTS:
            reject(
                callback.token,
                "budget.callback",
                "callback array exceeds 4096 captured effects",
            )
        captured.append(effects)
    return tuple(captured)


def capture_effects(
    owner: _Adapt, token: Token, statements: tuple[Statement, ...]
) -> tuple[Operation, ...]:
    """Compile a lexical body against a snapshot of its borrowed locations."""
    location = token.location
    syntax = Function(
        Token(f"$capture_{location.line}_{location.column}", location),
        tuple(owner.parameters.values()),
        statements,
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
    child.classical_matrices = dict(owner.classical_matrices)
    child.zero_callbacks = dict(owner.zero_callbacks)
    child.classical_integers = dict(owner.classical_integers)
    child.integer_arrays = dict(owner.integer_arrays)
    child.indexed_pauli = dict(owner.indexed_pauli)
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
    for ordinal, statement in enumerate(statements):
        child.statement(statement, str(ordinal))
    if (
        any(op.name not in PURE_EFFECTS for op in child.operations)
        or child.functions
        or child.registers != owner.registers
        or child.views != owner.views
        or child.suspended != owner.suspended
    ):
        reject(
            token,
            "callback.escape",
            "lexical callback requires pure effects on existing storage",
        )
    owner.helper_expansions = child.helper_expansions
    owner.loop_iterations = child.loop_iterations
    return tuple(child.operations)


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

    def phase(locations: tuple[tuple[Value, int], ...], angle: float) -> None:
        selected = (*controls, *locations)
        if len(selected) > 8 or len(set(selected)) != len(selected):
            reject(
                token,
                "callback.control",
                "selected phase needs at most eight distinct bits",
            )
        owner.effect_op(
            "qreg.control_phase",
            token,
            tuple(value for value, _ in selected),
            indices=tuple(bit for _, bit in selected),
            angle=angle,
        )

    for operation in effects:
        operands = operation.operands[1:]
        attributes = operation.attributes
        if operation.name == "qreg.control_x" and "indices" in attributes:
            locations = tuple(zip(operands, attributes["indices"], strict=True))
            owner.emit_indexed_x(token, (*controls, *locations))
            continue
        if operation.name == "qreg.phase" and not operands:
            phase((), attributes["angle"])
            continue
        if operation.name == "qreg.control_phase" and "indices" in attributes:
            phase(
                tuple(zip(operands, attributes["indices"], strict=True)),
                attributes["angle"],
            )
            continue
        if (
            operation.name == "qreg.inplace_prepare_amplitudes"
            and "control_count" not in attributes
        ):
            targets = (
                tuple(zip(operands, attributes["indices"], strict=True))
                if "indices" in attributes
                else tuple(
                    (operands[0], bit)
                    for bit in reversed(range(register_width(operands[0])))
                )
            )
            selected = (*controls, *targets)
            if len(selected) > 8 or len(set(selected)) != len(selected):
                reject(
                    token,
                    "callback.control",
                    "selected preparation needs at most eight distinct bits",
                )
            owner.effect_op(
                operation.name,
                token,
                tuple(value for value, _ in selected),
                **{
                    **attributes,
                    "indices": tuple(bit for _, bit in selected),
                    "control_count": len(controls),
                },
            )
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
        elif gate in {"h", "rx", "ry"}:
            angle = -math.pi / 4 if gate == "h" else attributes["angle"] / 2
            if attributes.get("inverse") and gate != "h":
                angle = -angle
            for direction in (1, -1):
                owner.effect_op(
                    "qreg.gate",
                    token,
                    (value,),
                    gate="ry" if gate == "h" else gate,
                    index=index,
                    angle=direction * angle,
                )
                if direction == 1 or gate != "h":
                    phase(((value, index),), math.pi)
        else:
            reject(
                token,
                "callback.control",
                "controlled callback scalar gate must be I, X, H, RX or RY",
            )
    for value, index in reversed(negative):
        owner.effect_op("qreg.gate", token, (value,), gate="x", index=index)
