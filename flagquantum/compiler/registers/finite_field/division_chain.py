# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix division and quotient erasure generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_div.cc

"""Binary-field division and measured erasure on fixed register wires."""

from ....core.classical_dataflow import ClassicalXor, parse_classical_xor
from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ..primitives.measurement import reset_x_register_ir
from ._physical_calls import append_field_call
from .inverse_chain import (
    field_inverse_chain_erase_ir,
    field_inverse_chain_ir,
    field_inverse_chain_size,
)
from .polynomial import _validate_registers
from .polynomial_product import append_field_product, field_product_xor_ir
from .product_erasure import field_product_erase_ir


def field_division_workspace_size(width: int) -> int:
    """Return the clean width for the inverse word and its chain."""
    return width + field_inverse_chain_size(width)


def _division_workspace(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = len(divisor)
    if len(dividend) != width or len(target) != width:
        raise ValueError("Division requires equal dividend, divisor, and target widths")
    if len(workspace) != field_division_workspace_size(width):
        raise ValueError("Division workspace must fit the inverse word and chain")
    # The two inputs are read in separate stages and may share wires.
    _validate_registers(dividend, target, workspace)
    _validate_registers(divisor, target, workspace)
    return workspace[:width], workspace[width:]


def field_division_chain_ir(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    modulus: int,
) -> tuple[Instruction, ...]:
    """XOR the quotient into target and keep the inverse chain live.

    Words are big-endian. The caller provides zeroed workspace, with the
    inverse word first. A zero divisor contributes a zero quotient.
    """
    inverse, chain = _division_workspace(dividend, divisor, target, workspace)
    return (
        *field_inverse_chain_ir(divisor, inverse, chain, modulus=modulus),
        *field_product_xor_ir(dividend, inverse, target, modulus=modulus),
    )


def field_division_chain_erase_ir(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    quotient: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Clear a known quotient and its live inverse chain without Toffolis.

    The caller supplies one outcome slot per quotient bit, most significant
    first, and a separate parity slot for fields that need mask folding.
    """
    inverse, chain = _division_workspace(dividend, divisor, quotient, workspace)
    return (
        *field_product_erase_ir(
            dividend,
            inverse,
            quotient,
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
        *field_inverse_chain_erase_ir(
            divisor,
            inverse,
            chain,
            measurement_bits if chain else measurement_bits[-1:],
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )


def field_division_xor_ir(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """XOR the quotient and return the initially clean workspace to zero.

    Supply one outcome slot for fields of degree one or two, and one per
    field bit otherwise, most significant first.
    """
    forward = field_division_chain_ir(
        dividend, divisor, target, workspace, modulus=modulus
    )
    width = len(divisor)
    return (
        *forward,
        *field_inverse_chain_erase_ir(
            divisor,
            workspace[:width],
            workspace[width:],
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )


def field_division_erase_ir(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    quotient: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bits: tuple[int, ...],
    *,
    modulus: int,
    parity_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Clear a known quotient, rebuilding its inverse chain in clean workspace.

    The result and parity slots follow the same contract as retained cleanup.
    """
    inverse, chain = _division_workspace(dividend, divisor, quotient, workspace)
    return (
        *field_inverse_chain_ir(divisor, inverse, chain, modulus=modulus),
        *field_division_chain_erase_ir(
            dividend,
            divisor,
            quotient,
            workspace,
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )


def physical_field_division(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    modulus: int,
    retain_chain: bool = False,
    erase: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the four source quotient routes and their physical call ranges.

    Each workspace word follows the big-endian operand binding. The retained
    erase route consumes the live chain, and all erase routes require a known
    quotient. The parity slot, when present, is initially zero.
    """
    width = len(divisor)
    required = field_division_workspace_size(width)
    active = workspace if retain_chain else workspace[:required]
    inverse, chain = _division_workspace(dividend, divisor, target, active)
    bits = tuple(reversed(range(width)))
    inverse_bits = bits if chain else bits[-1:]
    parity = width
    gates: list[Instruction] = []
    children: list[ComponentCall] = []
    if not (retain_chain and erase):
        for start in range(0, len(active), width):
            gates.extend(reset_x_register_ir(active[start : start + width][::-1]))
        calls: list[ComponentCall] = []
        forward = field_inverse_chain_ir(
            divisor,
            inverse,
            chain,
            modulus=modulus,
            calls=calls,
        )
        append_field_call(gates, children, "gf_inverse_chain", forward, tuple(calls))
    append_field_product(
        gates,
        children,
        dividend,
        inverse,
        target,
        modulus,
        bits=bits if erase else None,
        parity_bit=parity,
    )
    if erase or not retain_chain:
        calls = []
        cleanup = field_inverse_chain_erase_ir(
            divisor,
            inverse,
            chain,
            inverse_bits,
            modulus=modulus,
            parity_bit=parity,
            emit_resets=True,
            calls=calls,
        )
        append_field_call(gates, children, "gf_uninverse_chain", cleanup, tuple(calls))

    # The first clean-bit lease uses initially zero storage in the target
    # format. Later allocations emit the source allocator's explicit store.
    trace = PhysicalTrace()
    for index, gate in enumerate(gates):
        if gate.name == "classical_xor":
            assignment = parse_classical_xor(gate)
            assert assignment.target == parity
            gates[index] = ClassicalXor(
                (*assignment.sources, parity), parity
            ).instruction()
            trace.implicit_zero_bits.add(parity)
            break
    name = "gf_undiv" if erase else "gf_div"
    trace.calls.append(ComponentCall(name, 0, len(gates), children=tuple(children)))
    last_wire = max((*dividend, *divisor, *target, *workspace))
    return lower_physical_trace(
        tuple(gates), 0, last_wire, trace, reorder_feedback=False
    )
