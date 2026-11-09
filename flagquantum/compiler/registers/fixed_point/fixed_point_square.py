# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 SquareRealNumber supplies the model but has no decomposition.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/multiplication.py
# Executable construction: https://arxiv.org/pdf/2007.07391#page=73

"""Square one unsigned fraction using the odd and even D6 truncation rules."""

from collections.abc import Iterator

from ....core.ir import Instruction
from ..bits.bit_compression import weighted_columns_xor_ir


def _retained_terms(width: int) -> Iterator[tuple[int, int, int | None]]:
    """Yield output weight and one-based input positions from D24 and D28."""
    diagonal_stop = (width - 1) // 2
    pair_limit = width + int(width % 2 == 0)
    for high in range(1, width + 1):
        if high <= diagonal_stop:
            yield width - 2 * high, high, None
        for low in range(1, min(high - 1, pair_limit - high) + 1):
            yield width + 1 - high - low, high, low


def fixed_point_square_workspace_size(width: int) -> int:
    """Count retained pair products and carries in the native network."""
    if type(width) is not int or width < 3:
        raise ValueError("Fixed-point square requires an input width of at least three")
    columns = [0] * width
    work = 0
    for position, _, low in _retained_terms(width):
        columns[position] += 1
        work += low is not None
    for i in range(width - 1):
        carries = columns[i] // 2
        columns[i + 1] += carries
        work += carries
    return work


def fixed_point_square_ir(
    source: tuple[int, ...],
    result: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
) -> tuple[Instruction, ...]:
    """XOR the D6 approximation into an equal-width fractional result.

    Both words are big endian with every bit fractional. The source is
    preserved. A zero result implements compute; an already computed
    result implements erase. All workspace finishes clean.
    """
    width = len(source)
    size = fixed_point_square_workspace_size(width)
    if len(result) != width or len(workspace) != size:
        raise ValueError(
            f"Fixed-point square requires {width} result and {size} work wires"
        )
    return weighted_columns_xor_ir(
        source,
        result,
        workspace,
        measurement_bit,
        (
            (column, source[high - 1], source[low - 1] if low is not None else None)
            for column, high, low in _retained_terms(width)
        ),
    )
