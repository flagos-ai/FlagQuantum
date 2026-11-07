# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix gen_gf_phase_by_product and gen_gf_unmul.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_mul.cc

"""Kickmix-ordered product phases and measurement-assisted product erasure."""

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction
from .field_erasure import _field_product_phase_masks
from .field_multiplication import _validate_field
from .measurement import ParityPhase, measure_x_clear_ir
from .polynomial import _validate_registers


def kickmix_field_product_phase_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    mask_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Apply the product's mod-two inner product with a classical mask as phase.

    Quantum words and mask slots are big-endian. For a multi-bit column, the
    caller lends a classical parity slot disjoint from the mask. Each such
    group restores that slot to zero. No quantum workspace is needed.
    """
    width = len(left)
    _validate_field(width, modulus)
    _validate_registers(left, right)
    if len(right) != width or len(mask_bits) != width:
        raise ValueError("Product phase requires equal register and mask widths")
    slots = mask_bits if parity_bit is None else (*mask_bits, parity_bit)
    if len(set(slots)) != len(slots) or any(
        type(bit) is not int or bit < 0 for bit in slots
    ):
        raise ValueError("Product phase requires distinct nonnegative classical slots")

    phases, high_masks = _field_product_phase_masks(left, right, mask_bits, modulus)
    offset = width * (width + 1) // 2
    gates = [gate for phase in phases[:offset] for gate in phase.lower()]
    for i, mask in enumerate(high_masks[:-1]):
        end = offset + width - i - 1
        group = phases[offset:end]
        offset = end
        if len(mask) <= 1:
            for phase in group:
                gates.extend(phase.lower())
            continue
        if parity_bit is None:
            raise ValueError(
                "Product phase needs a classical parity slot for multi-bit masks"
            )
        # The IR initializes even the first group explicitly. Kickmix's
        # first allocation obtains an implicitly zero classical bit.
        gates.append(ClassicalXor(mask, parity_bit).instruction())
        for phase in group:
            gates.extend(ParityPhase(phase.qubits, (parity_bit,)).lower())
        gates.append(ClassicalXor((*mask, parity_bit), parity_bit).instruction())
    return tuple(gates)


def kickmix_field_product_erase_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    product: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Clear a known product on every X-measurement branch, preserving its inputs.

    Product must hold left*right over the supplied field. Outcome slots are
    big-endian and remain caller-owned. Each consumed physical product wire
    returns in zero. The operation uses no additional quantum work.
    """
    _validate_registers(left, right, product)
    if len(product) != len(left):
        raise ValueError("Product erasure requires equal register widths")
    feedback = kickmix_field_product_phase_ir(
        left, right, measurement_bits, modulus=modulus, parity_bit=parity_bit
    )
    return (
        *(
            gate
            for wire, bit in zip(product[::-1], measurement_bits[::-1], strict=True)
            for gate in measure_x_clear_ir(wire, bit)
        ),
        *feedback,
    )
