# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_cmp_low_space.cc and CircuitBuilder::cccz.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators/gen_cmp_low_space.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Apply exact masked prefix phases with caller-owned clean and dirty wires."""

from ....core.ir import Instruction
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..primitives._mixed_gates import _flip
from ..primitives.measurement import measure_x_clear_ir


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
    if clean is None:
        # The source borrows the lowest wire ID outside this gate's operands.
        helper = next(q for q in range(5) if q not in (first, second, third, mask))
        return (*_flip(helper, first, second), *_flip(None, helper, third, mask)) * 2
    if measurement_bit is None:
        raise ValueError("Masked prefix phase requires a classical result slot")
    return (
        *_flip(clean, first, second),
        *_flip(None, clean, third, mask),
        *measure_x_clear_ir(clean, measurement_bit),
        *_flip(None, first, second, ClassicalFieldCoefficient(measurement_bit)),
    )


def kickmix_masked_prefix_phase_ir(
    targets: tuple[int, ...],
    masks: tuple[int | bool | ClassicalFieldCoefficient, ...],
    *,
    clean: tuple[int, ...] = (),
    dirty: tuple[int, ...] = (),
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
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
