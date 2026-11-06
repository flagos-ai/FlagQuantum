"""Probability preparation on initialized Qmod storage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from .amplification import checked_probabilities, product_rotation_angles
from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def lower_inplace_preparation(
    owner: _Adapt, token: Token, arguments: tuple[Expression, ...]
) -> None:
    if len(arguments) != 3:
        reject(
            token,
            "signature",
            "inplace preparation requires probabilities, bound and target",
        )
    vector, bound, target = arguments
    probabilities = checked_probabilities(vector, bound, owner.constants)
    if target.children or target.token.text in owner.views:
        locations = tuple(reversed(owner.array_view(target)))
        if len(probabilities) != (1 << len(locations)):
            reject(token, "width", "probability vector width disagrees")
        owner.effect_op(
            "qreg.inplace_prepare_amplitudes",
            token,
            tuple(value for value, _ in locations),
            indices=tuple(index for _, index in locations),
            amplitudes=tuple(math.sqrt(value) for value in probabilities),
            bound=0.0,
        )
        return
    register = owner.register(owner.atom(target), allow_qubit_array=True)
    if len(probabilities) != (1 << register_width(register)):
        reject(token, "width", "probability vector width disagrees")
    angles = product_rotation_angles(probabilities)
    if angles is None:
        owner.effect_op(
            "qreg.inplace_prepare_amplitudes",
            token,
            (register,),
            amplitudes=tuple(math.sqrt(value) for value in probabilities),
            bound=0.0,
        )
        return
    for bit, angle in enumerate(angles):
        owner.effect_op(
            "qreg.gate", token, (register,), gate="ry", index=bit, angle=angle
        )
