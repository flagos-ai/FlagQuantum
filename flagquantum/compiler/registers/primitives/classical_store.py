# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix classical store and Python broadcast binding.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops

"""Bind ordered classical stores to the existing physical opcode artifact."""

from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .broadcast_binding import broadcast_operand_words


def physical_classical_store(
    target: object, value: object, *, control: object = True
) -> ReversiblePhaseProgram:
    """Store a Boolean in each enabled classical lane, retaining sequential aliases.

    The caller owns all target and control slots. A scalar port broadcasts over
    the first explicit word. Conditions read their value at each store, so a
    preceding lane can change a following lane's enable. The one-wire artifact
    minimum supplies an untouched spectator for the existing append path.
    """
    if value is not None and getattr(type(value), "__bool__", None) is None:
        raise ValueError("Classical store value requires a Boolean-convertible scalar")
    try:
        literal = bool(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Classical store value requires a Boolean-convertible scalar"
        ) from error
    targets, controls = broadcast_operand_words(target, control)
    if any(not isinstance(port, ClassicalFieldCoefficient) for port in targets):
        raise ValueError("Classical store targets must be classical bits")
    if any(
        type(port) is not bool and not isinstance(port, ClassicalFieldCoefficient)
        for port in controls
    ):
        raise ValueError("Classical store control must be classical")
    slots = tuple(
        sorted(
            {
                port.slot
                for port in (*targets, *controls)
                if isinstance(port, ClassicalFieldCoefficient)
            }
        )
    )
    operations = []
    for output, enable in zip(targets, controls, strict=True):
        assert isinstance(output, ClassicalFieldCoefficient)
        if enable is False:
            continue
        operations.append(
            Operation(
                "bit_store1" if literal else "bit_store0",
                (),
                result_bit=output.slot,
                condition_bit=(
                    enable.slot
                    if isinstance(enable, ClassicalFieldCoefficient)
                    else None
                ),
            )
        )
    return ReversiblePhaseProgram(
        1, max(slots, default=-1) + 1, tuple(operations), slots
    )
