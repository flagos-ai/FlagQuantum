"""Closed preparation and phase-exact Grover reflection on borrowed storage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .amplification import callback
from .syntax import Expression, Token, reject
from .unitary_effects import emit_inverse

if TYPE_CHECKING:
    from .context import _Adapt


def lower_diffuser(owner: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    if len(args) != 2:
        reject(token, "signature", "grover_diffuser requires a preparation and target")
    preparation, target = args
    locations = owner.array_view(target)
    width = len(locations)
    if not 1 <= width <= 8 or len(set(locations)) != width:
        reject(
            token, "diffusion.width", "diffusion requires one to eight distinct bits"
        )
    body = callback(preparation, target, width, owner.definitions)
    if len(body) != 1 or body[0].control_body is not None:
        reject(
            token, "diffusion.preparation", "diffusion requires one closed preparation"
        )
    operation = body[0]
    arguments = operation.arguments
    gate = operation.token.text
    start, incoming = len(owner.operations), owner.effect
    if (
        gate == "hadamard_transform"
        and arguments == (target,)
        or gate == "prepare_uniform_trimmed_state"
        and len(arguments) == 2
        and arguments[1] == target
        and not arguments[0].children
        and arguments[0].token.text == str(1 << width)
    ):
        for value, wire in locations:
            owner.effect_op("qreg.gate", token, (value,), gate="h", index=wire)
    elif (
        gate == "inplace_prepare_state"
        and len(arguments) == 3
        and arguments[2] == target
    ):
        owner.statement(operation, "diffusion")
    else:
        reject(
            token,
            "diffusion.preparation",
            "unsupported or capturing diffusion preparation",
        )
    forward = tuple(owner.operations[start:])
    del owner.operations[start:]
    owner.effect = incoming
    emit_inverse(owner, token, forward)
    for value, wire in locations:
        owner.effect_op("qreg.gate", token, (value,), gate="x", index=wire)
    owner.effect_op(
        "qreg.control_phase",
        token,
        tuple(value for value, _ in locations),
        indices=tuple(wire for _, wire in locations),
        angle=math.pi,
    )
    for value, wire in locations:
        owner.effect_op("qreg.gate", token, (value,), gate="x", index=wire)
    owner.effect_op("qreg.phase", token, (), angle=math.pi)
    for op in forward:
        owner.effect_op(op.name, token, op.operands[1:], **op.attributes)
