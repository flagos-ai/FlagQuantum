# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Zero-based arithmetic contract from Qualtran v0.7.0 ToContiguousIndex:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/contiguous_index.py

"""Zero-based triangular indexing with shared square and add circuits."""

from collections.abc import Iterator

from ...core.ir import Instruction
from .arithmetic import add_adjoint_ir, add_ir
from .bit_compression import with_weighted_columns_ir
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


def _triangular_terms(width: int) -> Iterator[tuple[int, int, int]]:
    """Yield (column, little-endian nu bit, other nu bit) for pair products."""
    for i in range(1, width):
        for j in range(i):
            yield i + j, i, j


def contiguous_index_columns_workspace_size(width: int) -> int:
    """Return work for Appendix F's linear copies, products and carries."""
    products = integer_square_workspace_size(width) // 2
    counts = [1] * width + [0] * width
    counts[0] += 1
    for i in range(1, width):
        counts[i - 1] += 1
        counts[2 * i - 1] += 1
    for column, _, _ in _triangular_terms(width):
        counts[column] += 1
    carries = 0
    for i in range(len(counts) - 1):
        count = counts[i] // 2
        carries += count
        counts[i + 1] += count
    return width - 1 + products + carries


def contiguous_index_columns_ir(
    mu: tuple[int, ...],
    nu: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR the index using direct weighted columns and one classical slot.

    Each linear occurrence has a distinct qubit because the compressor
    temporarily mutates parity operands. The work pool is restored.
    """
    width = len(nu)
    required = contiguous_index_columns_workspace_size(width)
    if len(mu) != width or not result:
        raise ValueError(
            "Contiguous index requires matching inputs and nonempty result"
        )
    if len(workspace) != required:
        raise ValueError(f"Contiguous index columns require {required} work qubits")
    _validate_wires((*mu, *nu, *result, *workspace))

    p = nu[::-1]
    copies = workspace[: width - 1]
    bind = xor_ir(p[1:], copies) if copies else ()
    # Appendix F, Eq. F6/F7: https://arxiv.org/pdf/2011.03494#page=68
    terms: list[tuple[int, int, int | None]] = [
        (i, qubit, None) for i, qubit in enumerate(reversed(mu))
    ]
    terms.append((0, p[0], None))
    for i, copy in enumerate(copies, start=1):
        terms.extend(((i - 1, p[i], None), (2 * i - 1, copy, None)))
    terms.extend((column, p[i], p[j]) for column, i, j in _triangular_terms(width))
    copied = min(len(result), 2 * width)

    def consume(word: tuple[int | None, ...]) -> tuple[Instruction, ...]:
        return tuple(
            Instruction("cx", (qubit, target))
            for qubit, target in zip(word[-copied:], result[-copied:], strict=True)
            if qubit is not None
        )

    return (
        *bind,
        *with_weighted_columns_ir(
            2 * width, workspace[width - 1 :], measurement_bit, terms, consume
        ),
        *bind,
    )
