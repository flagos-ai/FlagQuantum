# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_cmp_low_space.cc and CircuitBuilder::cccz.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators/gen_cmp_low_space.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Apply exact masked prefix phases with caller-owned clean and dirty wires."""

from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip, _three_control_ir
from ..primitives.measurement import measure_x_clear_ir, reset_x_register_ir


def _four_operand_phase(
    first: int,
    second: int,
    third: int,
    mask: int | bool | ClassicalFieldCoefficient,
    clean: int | None = None,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    if isinstance(mask, (bool, ClassicalFieldCoefficient)):
        return _flip(None, first, second, third, mask)
    if clean is not None and measurement_bit is None:
        raise ValueError("Masked prefix phase requires a classical result slot")
    return _three_control_ir(
        first,
        second,
        third,
        mask,
        phase=True,
        scratch=clean,
        measurement_bit=measurement_bit,
    )


def masked_prefix_phase_ir(
    targets: tuple[int, ...],
    masks: tuple[int | bool | ClassicalFieldCoefficient, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
    emit_resets: bool = False,
) -> tuple[Instruction, ...]:
    """Apply one sign per all-ones prefix whose corresponding mask is set.

    Use the common length of targets and masks, starting at index zero.
    A mask is a constant, an existing classical slot, or a quantum wire.
    Carry storage uses the caller's clean wires before dirty wires, in tuple
    order. Clean inputs start at zero and all workspace is restored. The
    borrowed four-operand phase uses the source's lowest available wire ID,
    which must be present in the containing circuit. The caller owns one
    reusable result slot when the selected schedule uses measurements.
    """
    n = min(len(targets), len(masks))
    if n == 0:
        return ()
    masks = masks[:n]
    quantum_masks = tuple(dict.fromkeys(mask for mask in masks if type(mask) is int))
    _validate_registers(targets, quantum_masks, clean, dirty)
    if len(clean) + len(dirty) < n - 3:
        raise ValueError("Masked prefix phase requires at least n - 3 work wires")
    if measurement_bit is not None and (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or ClassicalFieldCoefficient(measurement_bit) in masks
    ):
        raise ValueError(
            "Masked prefix phase requires a separate classical result slot"
        )

    gates = list(_flip(None, targets[0], masks[0]))
    if n <= 3:
        if n >= 2:
            gates.extend(_flip(None, targets[0], targets[1], masks[1]))
        if n == 3:
            if emit_resets and clean and not minimize_qubits and type(masks[2]) is int:
                gates.extend(reset_x_register_ir((clean[0],)))
            gates.extend(
                _four_operand_phase(
                    targets[0],
                    targets[1],
                    targets[2],
                    masks[2],
                    clean[0] if clean and not minimize_qubits else None,
                    measurement_bit,
                )
            )
        return tuple(gates)

    anc = (*clean, *dirty)
    if clean and measurement_bit is None:
        raise ValueError("Masked prefix phase requires a classical result slot")
    if emit_resets:
        gates.extend(reset_x_register_ir(clean[: n - 3]))
    last_mask = masks[n - 1]
    for step in range(2):
        for k in reversed(range(n - 4)):
            if step == 0 and k < len(clean):
                continue
            if step == 1 and k + 1 < len(clean):
                assert measurement_bit is not None
                gates.extend(measure_x_clear_ir(anc[k + 1], measurement_bit))
                gates.extend(
                    _flip(
                        None,
                        targets[k + 2],
                        anc[k],
                        ClassicalFieldCoefficient(measurement_bit),
                    )
                )
            else:
                gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        if step == 1 and clean:
            assert measurement_bit is not None
            gates.extend(measure_x_clear_ir(anc[0], measurement_bit))
            gates.extend(
                _flip(
                    None,
                    targets[0],
                    targets[1],
                    ClassicalFieldCoefficient(measurement_bit),
                )
            )
        else:
            gates.extend(_flip(anc[0], targets[0], targets[1]))
        if step == 0:
            for k in range(min(len(clean), n - 4)):
                gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        for k in range(len(clean), n - 4):
            gates.extend(_flip(anc[k + 1], targets[k + 2], anc[k]))
        for k in range(n - 3):
            if k >= len(clean) or step == 0:
                gates.extend(_flip(None, masks[k + 1], anc[k]))
        if len(clean) < n - 3 or step == 0:
            gates.extend(_flip(None, masks[n - 2], targets[n - 2], anc[n - 4]))
            gates.extend(
                _four_operand_phase(
                    last_mask, targets[n - 1], targets[n - 2], anc[n - 4]
                )
                if type(last_mask) is int
                else _flip(None, last_mask, targets[n - 1], targets[n - 2], anc[n - 4])
            )
    return tuple(gates)


def physical_masked_prefix_phase(
    targets: tuple[int, ...],
    masks: tuple[int | bool | ClassicalFieldCoefficient, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    minimize_qubits: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind the source prefix phase and preserve its physical workspace marks."""
    inputs = {
        mask.slot for mask in masks if isinstance(mask, ClassicalFieldCoefficient)
    }
    ports = (*targets, *masks, *clean, *dirty)
    last_wire = max((q for q in ports if type(q) is int), default=0)
    gates = masked_prefix_phase_ir(
        targets,
        masks,
        clean=clean,
        dirty=dirty,
        minimize_qubits=minimize_qubits,
        measurement_bit=max(inputs, default=-1) + 1,
        emit_resets=True,
    )
    trace = PhysicalTrace(
        calls=[
            ComponentCall("masked_phase_by_prefix_using_dirty_workspace", 0, len(gates))
        ],
        implicit_zero_bits=inputs,
    )
    return lower_physical_trace(gates, 0, last_wire, trace, reorder_feedback=False)
