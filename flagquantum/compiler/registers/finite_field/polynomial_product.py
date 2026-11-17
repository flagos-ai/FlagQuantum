# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix Karatsuba generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_mul.cc

"""Fixed-wire product accumulation with the pinned Kickmix Karatsuba schedule."""

from collections.abc import Iterator

from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ..primitives.measurement import reset_x_register_ir
from ._physical_calls import append_field_call, lower_field_erasure_trace
from .binary_linear_map import (
    append_field_constant_scale,
    append_field_monomial_scale,
)
from .coefficients import gf2_add_coefficients_ir, gf2_add_ir
from .field_multiplication import _validate_field
from .polynomial import _validate_registers
from .product_erasure import field_product_erase_ir


def _scaled_polynomial_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...], shift: int
) -> Iterator[Instruction]:
    """Add (1 + x**shift) times a product to an overlapping target window."""
    if len(left) == 1:
        copy = Instruction("cx", (target[shift], target[0]))
        yield copy
        yield Instruction("ccx", (left[0], right[0], target[shift]))
        yield copy
        return

    product_width = 2 * len(left) - 1
    overlap = max(0, product_width - shift)
    copied = min(shift, product_width)
    # The upper ladder can share wires between CNOTs. Apply it high to low
    # before accumulation and in reverse order afterward.
    upper = tuple(
        Instruction("cx", (target[2 * shift + i], target[shift + i]))
        for i in range(overlap)
    )
    lower = gf2_add_ir(target[shift : shift + copied], target[:copied])
    yield from reversed(upper)
    yield from lower
    yield from _polynomial_product(target[shift : shift + product_width], left, right)
    yield from lower
    yield from upper


def _polynomial_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...]
) -> Iterator[Instruction]:
    """Add a carryless polynomial product on little-endian wires."""
    width = len(left)
    if width == 1:
        yield Instruction("ccx", (left[0], right[0], target[0]))
        return

    half = (width + 1) // 2
    yield from _scaled_polynomial_product(
        target[: 3 * half - 1], left[:half], right[:half], half
    )
    yield from _scaled_polynomial_product(
        target[half:], left[half:], right[half:], half
    )
    yield from _middle_product(target[half : 3 * half - 1], left, right, half)


def _middle_product(
    target: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...], half: int
) -> Iterator[Instruction]:
    low_width = len(left) - half
    add_left = gf2_add_ir(left[half:], left[:low_width])
    add_right = gf2_add_ir(right[half:], right[:low_width])
    yield from add_left
    yield from add_right
    yield from _polynomial_product(target, left[:half], right[:half])
    yield from add_right
    yield from add_left


def polynomial_product_xor_ir(
    left: tuple[int, ...], right: tuple[int, ...], target: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """XOR a carryless product into a big-endian word without field reduction."""
    width = len(left)
    if width == 0:
        return ()
    if len(right) != width or len(target) != 2 * width - 1:
        raise ValueError("Polynomial product requires equal factors and a 2n-1 target")
    _validate_registers(left, right, target)
    return tuple(_polynomial_product(target[::-1], left[::-1], right[::-1]))


def physical_polynomial_product(
    left: tuple[int, ...], right: tuple[int, ...], target: tuple[int, ...]
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the unmarked source carryless product path."""
    body = polynomial_product_xor_ir(left, right, target)
    return lower_physical_trace(
        body,
        0,
        max((*left, *right, *target), default=0),
        PhysicalTrace(),
        reorder_feedback=False,
    )


def field_product_xor_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    *,
    modulus: int,
    calls: list[ComponentCall] | None = None,
) -> tuple[Instruction, ...]:
    """Add left*right into an arbitrary target over GF(2^m), on fixed wires.

    Registers are disjoint, equally wide, and big-endian. The two factors
    are preserved and no quantum or classical workspace is needed.
    """
    width = len(target)
    _validate_field(width, modulus)
    _validate_registers(left, right, target)
    if len(left) != width or len(right) != width:
        raise ValueError("Field product requires equal register widths")
    if width == 1:
        return (Instruction("ccx", (left[0], right[0], target[0])),)

    half = (width + 1) // 2
    high_width = width - half
    factor = 1 | (1 << half)
    x, y, result = left[::-1], right[::-1], target[::-1]
    gates: list[Instruction] = []
    append_field_monomial_scale(gates, calls, target, -half, modulus)
    gates.extend(_middle_product(result[: 2 * half - 1], x, y, half))
    append_field_constant_scale(gates, calls, target, factor, modulus, divide=True)
    gates.extend(_polynomial_product(result[: 2 * high_width - 1], x[half:], y[half:]))
    append_field_monomial_scale(gates, calls, target, half, modulus)
    gates.extend(_polynomial_product(result[: 2 * half - 1], x[:half], y[:half]))
    append_field_constant_scale(gates, calls, target, factor, modulus)
    return tuple(gates)


def append_field_product(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    first: tuple[int, ...],
    second: tuple[int, ...],
    target: tuple[int, ...],
    modulus: int,
    *,
    bits: tuple[int, ...] | None = None,
    parity_bit: int | None = None,
) -> None:
    """Append product accumulation or erasure with optional physical call ranges."""
    children: list[ComponentCall] | None = [] if calls is not None else None
    if bits is None:
        name = "gf_mul"
        body = field_product_xor_ir(
            first, second, target, modulus=modulus, calls=children
        )
    else:
        name = "gf_unmul"
        body = field_product_erase_ir(
            first,
            second,
            target,
            bits,
            modulus=modulus,
            parity_bit=parity_bit,
            calls=children,
        )
    append_field_call(gates, calls, name, body, tuple(children or ()))


def controlled_field_product_xor_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
    emit_resets: bool = False,
    calls: list[ComponentCall] | None = None,
) -> tuple[Instruction, ...]:
    """XOR the product into target under a positive quantum control.

    Big-endian workspace starts clean and returns to zero on every branch.
    The caller provides one result slot per field bit and a separate parity
    slot when the modulus needs one. The same schedule is its XOR inverse.
    """
    _validate_registers((control,), left, right, target, workspace)
    add = gf2_add_coefficients_ir(workspace, target, control=control)
    gates = list(reset_x_register_ir(workspace[::-1])) if emit_resets else []
    append_field_product(gates, calls, left, right, workspace, modulus)
    append_field_call(gates, calls, "c_gf_iadd", add)
    append_field_product(
        gates,
        calls,
        left,
        right,
        workspace,
        modulus,
        bits=measurement_bits,
        parity_bit=parity_bit,
    )
    return tuple(gates)


def physical_field_product(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    *,
    modulus: int,
    erase: bool = False,
    control: int | None = None,
    workspace: tuple[int, ...] = (),
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export accumulation or erasure with the source control dispatch."""
    width = len(left)
    bits = tuple(reversed(range(width)))
    calls: list[ComponentCall] = []
    ports = (*left, *right, *target, *workspace)
    if control is None:
        gates: list[Instruction] = []
        append_field_product(
            gates,
            calls,
            left,
            right,
            target,
            modulus,
            bits=bits if erase else None,
            parity_bit=width,
        )
        (call,) = calls
    else:
        gates = list(
            controlled_field_product_xor_ir(
                control,
                left,
                right,
                target,
                workspace[:width],
                bits,
                modulus=modulus,
                parity_bit=width,
                emit_resets=True,
                calls=calls,
            )
        )
        call = ComponentCall("c_gf_mul", 0, len(gates), children=tuple(calls))
        ports = (*ports, control)
    return lower_field_erasure_trace(gates, call, width, max(ports))
