# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix basis-aware control binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/control_helper.pybind.cc

"""Resolve typed control operands and apply their classical predicates."""

from dataclasses import replace

from ....core.ir import Instruction
from ...artifacts.operand_identity import OperandIdentity
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ._mixed_gates import _Control, _XBasisTarget
from .operand_binding import bind_operand


def bind_z_enable(
    control: _Control | OperandIdentity,
) -> tuple[int | bool, tuple[int, ...]]:
    """Return the quantum enable or literal and the outer classical reads."""
    operand = bind_operand(control)
    if isinstance(operand, ClassicalFieldCoefficient):
        return True, (operand.slot,)
    if isinstance(operand, int):
        return operand, ()
    raise ValueError("Control must be a Boolean, quantum wire or classical slot")


def bind_x_output(
    target: int | _XBasisTarget | OperandIdentity | str,
) -> tuple[int | None, bool, tuple[int, ...]]:
    """Return the target wire or phase, skip flag and outer classical reads."""
    operand = bind_operand(target)
    if type(operand) is int:
        return operand, False, ()
    if isinstance(operand, _XBasisTarget):
        if isinstance(operand.outcome, ClassicalFieldCoefficient):
            return None, False, (operand.outcome.slot,)
        if type(operand.outcome) is bool:
            return None, not operand.outcome, ()
    raise ValueError("Output must be a quantum wire or an X-basis operand")


def apply_control_guards(
    instructions: tuple[Instruction, ...], slots: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Conjoin read-only outer guards with an emitted conjunction-only body.

    An outer slot must remain unchanged across the body. The caller snapshots
    any slot the body will write and supplies a defined guarded result slot.
    """
    if not slots:
        return instructions
    required = {(slot, 1) for slot in slots}
    guarded = []
    for gate in instructions:
        metadata = {
            **gate.metadata,
            "conditions": tuple(
                sorted(required.union(gate.metadata.get("conditions", ())))
            ),
        }
        if gate.name in {"measure", "reset"}:
            metadata["guarded_effect"] = True
        guarded.append(replace(gate, metadata=metadata))
    return tuple(guarded)
