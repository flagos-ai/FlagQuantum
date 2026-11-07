# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix division and quotient erasure generators.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/gf_arithmetic/gen_gf_div.cc

"""Binary-field division and measured erasure on fixed register wires."""

from ...core.ir import Instruction
from .kickmix_field_erasure import kickmix_field_product_erase_ir
from .kickmix_field_inverse import (
    kickmix_field_inverse_chain_erase_ir,
    kickmix_field_inverse_chain_ir,
    kickmix_field_inverse_chain_size,
)
from .kickmix_field_product import kickmix_field_product_xor_ir
from .polynomial import _validate_registers


def kickmix_field_division_workspace_size(width: int) -> int:
    """Return the clean width for the inverse word and its chain."""
    return width + kickmix_field_inverse_chain_size(width)


def _division_workspace(
    dividend: tuple[int, ...],
    divisor: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    width = len(divisor)
    if len(dividend) != width or len(target) != width:
        raise ValueError("Division requires equal dividend, divisor, and target widths")
    if len(workspace) != kickmix_field_division_workspace_size(width):
        raise ValueError("Division workspace must fit the inverse word and chain")
    # The two inputs are read in separate stages and may share wires.
    _validate_registers(dividend, target, workspace)
    _validate_registers(divisor, target, workspace)
    return workspace[:width], workspace[width:]


def kickmix_field_division_chain_ir(
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
        *kickmix_field_inverse_chain_ir(divisor, inverse, chain, modulus=modulus),
        *kickmix_field_product_xor_ir(dividend, inverse, target, modulus=modulus),
    )


def kickmix_field_division_chain_erase_ir(
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
        *kickmix_field_product_erase_ir(
            dividend,
            inverse,
            quotient,
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
        *kickmix_field_inverse_chain_erase_ir(
            divisor,
            inverse,
            chain,
            measurement_bits if chain else measurement_bits[-1:],
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )


def kickmix_field_division_xor_ir(
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
    forward = kickmix_field_division_chain_ir(
        dividend, divisor, target, workspace, modulus=modulus
    )
    width = len(divisor)
    return (
        *forward,
        *kickmix_field_inverse_chain_erase_ir(
            divisor,
            workspace[:width],
            workspace[width:],
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )


def kickmix_field_division_erase_ir(
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
        *kickmix_field_inverse_chain_ir(divisor, inverse, chain, modulus=modulus),
        *kickmix_field_division_chain_erase_ir(
            dividend,
            divisor,
            quotient,
            workspace,
            measurement_bits,
            modulus=modulus,
            parity_bit=parity_bit,
        ),
    )
