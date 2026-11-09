# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 Square supplies the numerical and resource contracts.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2105.12767#page=76

"""Unsigned squares and their sum through reversible column compression."""

from collections.abc import Iterator

from ....core.ir import Instruction
from ..bits.bit_compression import weighted_columns_xor_ir


def integer_square_workspace_size(width: int) -> int:
    """Count the off-diagonal products and column carries."""
    if type(width) is not int or width < 1:
        raise ValueError("Integer square requires a positive input width")
    return width * (width - 1)


def _square_terms(width: int) -> Iterator[tuple[int, int, int]]:
    """Yield the two input bit indices and the output column for each term."""
    for i in range(width):
        yield i, i, 2 * i
        for j in range(i):
            yield i, j, i + j + 1


def sum_of_squares_requirements(width: int, count: int) -> tuple[int, int]:
    """Return the pinned result width and native work count."""
    products_and_carries = integer_square_workspace_size(width)
    if type(count) is not int or count < 1:
        raise ValueError("Sum of squares requires a positive register count")
    if count == 1:
        return 2 * width, products_and_carries
    result_width = 2 * width + (count - 1).bit_length()
    columns = [0] * result_width
    work = count * (products_and_carries // 2)
    for _, _, column in _square_terms(width):
        columns[column] += count
    for i in range(result_width - 1):
        carries = columns[i] // 2
        columns[i + 1] += carries
        work += carries
    return result_width, work


def integer_square_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR an unsigned square into a double-width result, preserving input.

    Registers are big endian. Starting with zero result implements Square.
    Starting with the computed square implements its adjoint on the valid
    subspace. The caller supplies clean workspace and one reusable slot for
    immediate measured erasure. Workspace returns to zero in either direction.
    """
    width = len(source)
    required = integer_square_workspace_size(width)
    if len(result) != 2 * width or len(workspace) != required:
        raise ValueError(
            f"Integer square requires {2 * width} result and {required} work wires"
        )
    return _emit_square_columns((source,), result, workspace, measurement_bit)


def sum_of_squares_ir(
    sources: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR the sum of squares into the result and restore inputs and work.

    All words are big endian. The same circuit erases a matching result.
    The caller supplies clean workspace and one reusable measurement slot.
    """
    if not sources or any(len(word) != len(sources[0]) for word in sources):
        raise ValueError("Sum of squares requires equal-width input words")
    result_width, work_width = sum_of_squares_requirements(
        len(sources[0]), len(sources)
    )
    if len(result) != result_width or len(workspace) != work_width:
        raise ValueError(
            f"Sum of squares requires {result_width} result and {work_width} work wires"
        )
    return _emit_square_columns(sources, result, workspace, measurement_bit)


def _emit_square_columns(
    sources: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bit: int,
) -> tuple[Instruction, ...]:
    return weighted_columns_xor_ir(
        tuple(wire for word in sources for wire in word),
        result,
        workspace,
        measurement_bit,
        (
            (position, word[-1 - i], word[-1 - j] if i != j else None)
            for word in sources
            for i, j, position in _square_terms(len(word))
        ),
    )
