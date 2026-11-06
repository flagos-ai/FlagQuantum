# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Zero-based arithmetic contract from Qualtran v0.7.0 ToContiguousIndex:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/contiguous_index.py

"""Zero-based triangular indexing with shared square and add circuits."""

from ...core.ir import Instruction
from .arithmetic import add_adjoint_ir, add_ir
from .bitwise import _validate_wires, xor_ir
from .integer_square import integer_square_ir, integer_square_workspace_size


def contiguous_index_requirements(width: int) -> tuple[int, int]:
    """Return clean workspace and reusable classical slots for two width-bit inputs."""
    square = integer_square_workspace_size(width)
    carries = 2 * width - 1
    return 2 * width + max(square, carries), carries


def contiguous_index_ir(
    mu: tuple[int, ...],
    nu: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR nu*(nu+1)//2 + mu into an unsigned big-endian result.

    The input registers have equal positive width and are preserved.
    A narrow result receives the low bits; a wider result is zero-extended.
    All initially clean workspace is restored. Repeating erases the mask.
    """
    width = len(nu)
    work, slots = contiguous_index_requirements(width)
    if len(mu) != width or not result:
        raise ValueError(
            "Contiguous index requires matching inputs and nonempty result"
        )
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Contiguous index requires {work} work qubits and {slots} slots"
        )
    if (
        any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError("Contiguous index requires distinct nonnegative slots")
    _validate_wires((*mu, *nu, *result, *workspace))

    numerator = workspace[: 2 * width]
    scratch = workspace[2 * width :]
    carries = scratch[:slots]
    square = integer_square_ir(
        nu,
        numerator,
        scratch[: integer_square_workspace_size(width)],
        measurement_bit=classical_bits[0],
    )
    # nu² + nu is even. Rebind the zero low bit as the leading quotient bit.
    quotient = (numerator[-1], *numerator[:-1])
    copied = min(len(quotient), len(result))
    return (
        *square,
        *add_ir(nu, numerator, carries, classical_bits),
        *add_ir(mu, quotient, carries, classical_bits),
        *xor_ir(quotient[-copied:], result[-copied:]),
        *add_adjoint_ir(mu, quotient, carries, classical_bits),
        *add_adjoint_ir(nu, numerator, carries, classical_bits),
        *square,
    )
