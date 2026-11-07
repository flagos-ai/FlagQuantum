"""Coherent controls on distinct scalar locations in statically resolved views."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
    from .context import _Adapt


def lower_indexed_control(
    owner: _Adapt, condition: Expression, operation: Statement
) -> bool:
    gate = operation.token.text
    rotations = {"RX", "RY", "RZ"}
    if gate not in {"X", "Y", "H"} | rotations or len(operation.arguments) != (
        2 if gate in rotations else 1
    ):
        return False

    controls: list[tuple[Value, int]]
    if condition.token.text == "&":
        controls = []
        expected = 0
        pending = [condition]
        while pending:
            operand = pending.pop()
            if operand.token.text == "&":
                pending.extend(reversed(operand.children))
                continue
            if operand.token.text != "==":
                return False
            source, literal = operand.children
            parameter = owner.parameters.get(source.token.text)
            if (
                source.children
                or parameter is None
                or parameter.kind not in {"qnum", "qbit"}
                or literal.children
                or not literal.token.text.isdecimal()
            ):
                return False
            bits = owner.array_view(source)
            pattern = owner.static_integer(literal)
            if not 0 <= pattern < 1 << len(bits):
                reject(
                    literal.token,
                    "control",
                    "control equality value must fit its register",
                )
            expected |= pattern << len(controls)
            controls.extend(bits)
            if len(controls) > 8:
                reject(
                    condition.token,
                    "control",
                    "indexed control exceeds eight input bits",
                )
        if len(set(controls)) != len(controls):
            return False
    elif condition.token.text in {"[", "and"}:
        pending = [condition]
        controls = []
        while pending:
            operand = pending.pop()
            if operand.token.text == "and":
                pending.extend(reversed(operand.children))
            else:
                if operand.children and operand.token.text not in {"[", "."}:
                    return False
                controls.append(owner.gate_target(operand))
                if len(controls) > 8:
                    reject(
                        condition.token,
                        "control",
                        "indexed control exceeds eight input bits",
                    )
        expected = (1 << len(controls)) - 1
    elif condition.token.text == "==":
        source, literal = condition.children
        if source.children or (
            gate in {"H", "RY"} and source.token.text not in owner.views
        ):
            return False
        parameter = owner.parameters.get(source.token.text)
        if parameter is None or parameter.kind != "qnum":
            return False
        controls = list(owner.array_view(source))
        expected = owner.static_integer(literal)
        if not 0 <= expected < (1 << len(controls)):
            reject(
                literal.token, "control", "control equality value must fit its register"
            )
        if len(controls) > 8:
            reject(
                condition.token, "control", "indexed control exceeds eight input bits"
            )
    else:
        return False

    target = owner.gate_target(
        operation.arguments[-1],
        allow_numeric=gate in {"RX", "RZ"} or condition.token.text in {"&", "=="},
    )
    locations = (*controls, target)
    token = operation.token
    if len(set(locations)) != len(locations):
        reject(token, "control", "controls and target must be distinct qubit locations")
    negative = [
        location
        for index, location in enumerate(controls)
        if not (expected >> index) & 1
    ]
    for register, bit in negative:
        owner.effect_op("qreg.gate", token, (register,), gate="x", index=bit)

    if gate in {"X", "Y"}:
        if gate == "Y":
            owner.effect_op(
                "qreg.gate",
                token,
                (target[0],),
                gate="s",
                index=target[1],
                inverse=True,
            )
        owner.emit_indexed_x(token, locations)
        if gate == "Y":
            owner.effect_op("qreg.gate", token, (target[0],), gate="s", index=target[1])
    else:
        register, index = target
        angle = (
            -math.pi / 4
            if gate == "H"
            else owner.angle(operation.arguments[0], token) / 2
        )

        def rotation(value: float) -> None:
            owner.effect_op(
                "qreg.gate",
                token,
                (register,),
                gate="ry" if gate == "H" else gate.lower(),
                index=index,
                angle=value,
            )

        def conjugate() -> None:
            if gate == "RZ":
                owner.emit_indexed_x(token, locations)
                return
            owner.effect_op(
                "qreg.control_phase",
                token,
                tuple(value for value, _ in locations),
                indices=tuple(bit for _, bit in locations),
                angle=math.pi,
            )

        # The active branch conjugates the middle rotation by an anticommuting Pauli.
        # Hadamard is the special case RY(pi/4) Z RY(-pi/4).
        rotation(angle)
        conjugate()
        rotation(-angle)
        if gate != "H":
            conjugate()
    for register, bit in reversed(negative):
        owner.effect_op("qreg.gate", token, (register,), gate="x", index=bit)
    return True


def lower_zero_reflection_control(
    owner: _Adapt, condition: Expression, operation: Statement
) -> bool:
    if operation.token.text != "reflect_about_zero":
        return False
    token = operation.token
    if len(operation.arguments) != 1:
        reject(token, "signature", "reflect_about_zero requires one quantum array")
    positive = True
    selector = condition
    if condition.token.text == "==":
        selector, literal = condition.children
        expected = owner.static_integer(literal)
        if expected not in {0, 1}:
            reject(
                literal.token,
                "control",
                "reflection selector must compare against zero or one",
            )
        positive = bool(expected)
    control, bit = owner.gate_target(selector)
    data = owner.array_view(operation.arguments[0])
    locations = ((control, bit), *data)
    if len(set(locations)) != len(locations) or len(data) > 8:
        reject(
            token,
            "control",
            "reflection requires distinct locations and at most eight data bits",
        )
    if not positive:
        owner.effect_op("qreg.gate", token, (control,), gate="x", index=bit)
    owner.effect_op("qreg.gate", token, (control,), gate="z", index=bit)
    for register, index in data:
        owner.effect_op("qreg.gate", token, (register,), gate="x", index=index)
    owner.effect_op(
        "qreg.control_phase",
        token,
        tuple(register for register, _ in locations),
        indices=tuple(index for _, index in locations),
        angle=math.pi,
    )
    for register, index in reversed(data):
        owner.effect_op("qreg.gate", token, (register,), gate="x", index=index)
    if not positive:
        owner.effect_op("qreg.gate", token, (control,), gate="x", index=bit)
    return True


def lower_nested_indexed_control(
    owner: _Adapt, condition: Expression, operation: Statement
) -> bool:
    """Combine equality guards around one gate while preserving their basis values."""
    if condition.token.text != "==":
        return False
    conjunction, leaf = condition, operation
    while leaf.control_body is not None:
        if (
            len(leaf.control_body) != 1
            or leaf.control_else is not None
            or len(leaf.arguments) != 1
            or leaf.arguments[0].token.text != "=="
        ):
            return False
        conjunction = Expression(
            Token("&", leaf.token.location), (conjunction, leaf.arguments[0])
        )
        leaf = leaf.control_body[0]
    return lower_indexed_control(owner, conjunction, leaf)


def lower_guarded_field_increment(owner: _Adapt, statement: Statement) -> bool:
    body = statement.control_body
    if body is None or len(body) != 1 or statement.control_else is not None:
        return False
    update = body[0]
    if update.control_body is not None:
        return _lower_field_increment_branches(owner, statement, update)
    if update.token.text != "+=" or update.arguments[0].token.text != ".":
        return False
    target, amount = update.arguments
    pending = [statement.arguments[0]]
    selected: list[tuple[Value, int]] = []
    negative: list[tuple[Value, int]] = []
    while pending:
        guard = pending.pop()
        if guard.token.text in {"and", "&"}:
            pending.extend(guard.children)
            continue
        if guard.token.text != "==":
            return False
        word, literal = guard.children
        field = owner.record_field(word)
        if field is None or field[0].kind not in {"qbit", "qnum"}:
            return False
        expected = owner.static_integer(literal)
        view = field[1]
        if not 0 <= expected < 1 << len(view):
            reject(
                literal.token, "control", "control equality value must fit its field"
            )
        selected.extend(view)
        negative.extend(
            location for i, location in enumerate(view) if not (expected >> i) & 1
        )
        if len(selected) > 8:
            reject(guard.token, "control", "field control exceeds eight input bits")
    target_locations = owner.array_view(target)
    if len(set((*selected, *target_locations))) != len(selected) + len(
        target_locations
    ):
        reject(target.token, "alias", "field controls and target must be disjoint")
    for value, index in negative:
        owner.effect_op("qreg.gate", update.token, (value,), gate="x", index=index)
    owner.increment(update.token, target, amount, controls=tuple(selected))
    for value, index in reversed(negative):
        owner.effect_op("qreg.gate", update.token, (value,), gate="x", index=index)
    return True


def _lower_field_increment_branches(
    owner: _Adapt, outer: Statement, inner: Statement
) -> bool:
    """Select literal updates on the disjoint branches of a scalar record flag."""
    if len(inner.arguments) != 1 or inner.control_body is None:
        return False
    selector = inner.arguments[0]
    field = owner.record_field(selector)
    if field is None or field[0].kind != "qbit":
        return False
    branches = [(1, inner.control_body)]
    if inner.control_else is not None:
        branches.append((0, inner.control_else))
    if any(
        len(body) != 1
        or body[0].token.text != "+="
        or body[0].arguments[0].token.text != "."
        for _, body in branches
    ):
        return False
    for state, body in branches:
        position = selector.token.location
        flag = Expression(
            Token("==", position), (selector, Expression(Token(str(state), position)))
        )
        condition = Expression(Token("&", position), (outer.arguments[0], flag))
        if not lower_guarded_field_increment(
            owner, replace(outer, arguments=(condition,), control_body=body)
        ):
            return False
    return True


def lower_constant_phase_guard(owner: _Adapt, statement: Statement) -> bool:
    """Keep an exact constant phase under nested indexed equality guards."""
    chain = []
    leaf = statement
    while leaf.control_body is not None:
        if (
            len(leaf.control_body) != 1
            or leaf.control_else is not None
            or len(leaf.arguments) != 1
        ):
            return False
        condition = leaf.arguments[0]
        if condition.token.text != "==" or len(condition.children) != 2:
            return False
        source, expected = condition.children
        if expected.children or not expected.token.text.isdecimal():
            return False
        chain.append((source, expected))
        leaf = leaf.control_body[0]
    if (
        not chain
        or leaf.token.text != "phase"
        or len(leaf.arguments) != 1
        or leaf.arguments[0].children
        or leaf.arguments[0].token.text != "pi"
        or "pi" in owner.parameters
        or len(chain) == 1
        and not chain[0][0].children
        and not (
            owner.helper_stack and (owner.widths.get(chain[0][0].token.text) or 0) > 1
        )
    ):
        return False
    locations: dict[tuple[Value, int], int] = {}
    consistent = True
    for source, expected in chain:
        view = (
            (owner.gate_target(source),)
            if source.token.text == "[" and not source.children[0].children
            else owner.array_view(source)
        )
        word = owner.static_integer(expected)
        if not 0 <= word < 1 << len(view):
            reject(expected.token, "control", "phase equality must fit its word")
        for index, location in enumerate(view):
            bit = word >> index & 1
            if location in locations and locations[location] != bit:
                consistent = False
            locations[location] = bit
    if not 1 <= len(locations) <= 8:
        reject(
            statement.token, "control", "phase equality guards need at most eight bits"
        )
    values, indices = zip(*locations, strict=True)
    owner.effect_op(
        "qreg.control_phase",
        statement.token,
        values,
        indices=indices,
        control_state=sum(bit << index for index, bit in enumerate(locations.values())),
        angle=math.pi if consistent else 0.0,
    )
    return True
