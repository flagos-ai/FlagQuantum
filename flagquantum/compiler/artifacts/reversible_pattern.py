# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix CircuitPattern row gathering.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_pattern.cc

"""Bind one reversible-phase operation pattern across physical lane columns."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TypeAlias

from .reversible_phase import ReversiblePhaseOperation

LaneColumn: TypeAlias = int | Sequence[int]


def _bind_column(columns: Sequence[LaneColumn], index: int, lane: int) -> int:
    if type(index) is not int or not 0 <= index < len(columns):
        raise ValueError("Reversible pattern operand has no bound column")
    column = columns[index]
    if isinstance(column, int):
        value = column
    else:
        if not 0 <= lane < len(column):
            raise ValueError("Reversible pattern lane is outside its column")
        value = column[lane]
    if type(value) is not int or value < 0:
        raise ValueError("Reversible pattern columns require nonnegative indices")
    return value


def bind_reversible_pattern(
    operations: Sequence[ReversiblePhaseOperation],
    lanes: range,
    *,
    qubits: Sequence[LaneColumn] = (),
    bits: Sequence[LaneColumn] = (),
) -> tuple[ReversiblePhaseOperation, ...]:
    """Bind local operands in lane order, preserving operation order within lanes.

    Integer columns repeat a fixed operand. Sequences specify physical lanes
    and may be sliced with either stride. Build the pattern once and pass a
    reversed range to reverse lane traversal.
    """
    bound = []
    for lane in lanes:
        if lane < 0:
            raise ValueError("Reversible pattern lanes require nonnegative indices")
        for operation in operations:
            bound.append(
                replace(
                    operation,
                    qubits=tuple(
                        _bind_column(qubits, index, lane) for index in operation.qubits
                    ),
                    result_bit=(
                        _bind_column(bits, operation.result_bit, lane)
                        if operation.result_bit is not None
                        else None
                    ),
                    condition_bit=(
                        _bind_column(bits, operation.condition_bit, lane)
                        if operation.condition_bit is not None
                        else None
                    ),
                )
            )
    return tuple(bound)
