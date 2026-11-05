"""Coherent controls on distinct scalar locations in statically resolved views."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .syntax import Expression, Statement, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def lower_indexed_control(
    owner: _Adapt, condition: Expression, operation: Statement
) -> bool:
    gate = operation.token.text
    if gate not in {"X", "H", "RY"} or len(operation.arguments) != (
        2 if gate == "RY" else 1
    ):
        return False

    if condition.token.text in {"[", "and"}:
        pending = [condition]
        controls = []
        while pending:
            operand = pending.pop()
            if operand.token.text == "and":
                pending.extend(reversed(operand.children))
            else:
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
        if source.children or (gate != "X" and source.token.text not in owner.views):
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

    target = owner.gate_target(operation.arguments[-1])
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

    if gate == "X":
        owner.emit_indexed_x(token, locations)
    else:
        register, index = target
        angle = (
            -math.pi / 4
            if gate == "H"
            else owner.angle(operation.arguments[0], token) / 2
        )

        def rotation(value: float) -> None:
            owner.effect_op(
                "qreg.gate", token, (register,), gate="ry", index=index, angle=value
            )

        def conditional_z() -> None:
            owner.effect_op(
                "qreg.control_phase",
                token,
                tuple(value for value, _ in locations),
                indices=tuple(bit for _, bit in locations),
                angle=math.pi,
            )

        # Conjugating Z gives H. Two Z factors reverse the middle RY on the active branch.
        rotation(angle)
        conditional_z()
        rotation(-angle)
        if gate == "RY":
            conditional_z()
    for register, bit in reversed(negative):
        owner.effect_op("qreg.gate", token, (register,), gate="x", index=bit)
    return True
