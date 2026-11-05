"""Coherent Fourier selection over whole initialized source registers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from .syntax import Statement, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def lower_fourier_control(owner: _Adapt, statement: Statement) -> bool:
    body = statement.control_body
    if body is None or len(body) != 1 or statement.control_else is not None:
        return False
    operation = body[0]
    inverse = operation.inverse_body is not None
    if operation.inverse_body is not None:
        if len(operation.inverse_body) != 1:
            return False
        operation = operation.inverse_body[0]
    gate = operation.token.text
    if gate not in {"qft", "IDENTITY"} or len(operation.arguments) != 1:
        return False
    if inverse and gate == "IDENTITY":
        return False

    token = statement.token
    condition = statement.arguments[0]
    attributes: dict[str, object] = {"inverse": True} if inverse else {}
    if condition.token.text == "==":
        selector, word = condition.children
        name = owner.atom(selector)
        parameter = owner.parameters.get(name.text)
        if (
            parameter is None
            or parameter.kind not in {"qnum", "qbit"}
            or name.text in owner.signed
            or name.text in owner.fractional_bits
        ):
            reject(
                name,
                "control",
                "Fourier selection requires an unsigned integer selector",
            )
        control = owner.register(name)
        attributes["control_value"] = owner.static_integer(word)
    else:
        if condition.children:
            reject(token, "control", "controlled QFT requires a scalar control port")
        control, bit = owner.gate_target(condition)
        if bit:
            reject(token, "control", "controlled QFT requires a scalar control port")

    target = owner.atom(operation.arguments[0])
    data = owner.register(target, allow_qubit_array=True)
    if owner.parameters[target.text].kind not in {"qbit[]", "qnum"}:
        reject(target, "type", "controlled QFT requires an array or integer register")
    if gate == "IDENTITY":
        # A zero phase still checks initialized selector and target locations.
        if control == data:
            reject(target, "control", "condition and target must be disjoint")
        word_value = attributes.get("control_value", 1)
        assert isinstance(word_value, int)
        if not 0 <= word_value < (1 << register_width(control)):
            reject(token, "control", "control equality value must fit its register")
        owner.effect_op(
            "qreg.control_phase", token, (control, data), indices=(0, 0), angle=0.0
        )
    else:
        owner.effect_op("qreg.control_qft", token, (control, data), **attributes)
    return True
