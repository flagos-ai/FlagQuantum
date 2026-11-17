"""Bind basis-aware output and enable operands to register equality."""

from ....core.ir import Instruction
from ..primitives._mixed_gates import _Control, _XBasisTarget
from ..primitives.control_binding import (
    apply_control_guards,
    bind_x_output,
    bind_z_enable,
)
from .mixed_comparison import quantum_equal_ir


def bound_quantum_equal_ir(
    left: tuple[int, ...],
    right: tuple[_Control, ...],
    target: int | _XBasisTarget,
    workspace: tuple[int, ...] = (),
    *,
    control: _Control = True,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Bind the equality child and preserve outer classical enable semantics."""
    wire, skip, target_slots = bind_x_output(target)
    enable, enable_slots = bind_z_enable(control)
    if skip or enable is False:
        return ()
    slots = (*target_slots, *enable_slots)
    if len(left) > 1 and measurement_bit in slots:
        raise ValueError("Equality feedback requires a separate result slot")
    body = quantum_equal_ir(
        left,
        right,
        wire,
        workspace,
        control=enable,
        measurement_bit=measurement_bit,
    )
    return apply_control_guards(body, slots)
