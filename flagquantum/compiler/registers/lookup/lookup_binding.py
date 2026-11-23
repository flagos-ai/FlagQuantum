# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix lookup.pybind.cc at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Bind full integer tables to existing lookup and measured-erasure kernels."""

from collections.abc import Sequence
from typing import SupportsInt, cast

from ..._component_trace.trace import ComponentCall, ComponentTrace
from ...artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from ..finite_field.polynomial import _validate_registers
from ..layout.operand_word import bind_quantum_word
from ..layout.register_bits import split_integer
from .measured_unlookup import measured_unlookup_operations
from .recursive_lookup import recursive_lookup_operations


def lookup_binding(
    table: Sequence[object],
    address: object,
    target: object,
    *,
    workspace: tuple[int, ...] = (),
    erase: bool = False,
    classical_width: int = 0,
    reused_phase_bits: tuple[int, ...] = (),
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Initialize or erase a complete table using caller-owned clean storage.

    Address and target words are MSB-first. Workspace lists the source clean
    leases in allocation order and supplies one qubit per address bit. Classical
    phase slots start at classical_width, followed by the reusable measurement
    slot. A reused phase prefix records allocator initialization explicitly.
    Empty targets accept any number of zero rows, as in the frozen wrapper.
    """
    selection = bind_quantum_word(address)
    output = bind_quantum_word(target)
    n, width = len(selection), len(output)
    if n > 32:
        raise ValueError("Lookup binding supports at most 32 address bits")
    if type(classical_width) is not int or classical_width < 0:
        raise ValueError("Classical storage width must be a nonnegative integer")
    _validate_registers(selection, output, workspace)
    if len(workspace) < n:
        raise ValueError("Lookup binding needs one clean lease per address bit")
    converted_rows = []
    for index in range(len(table)):
        value = int(cast(SupportsInt, table[index]))
        if not 0 <= value < 1 << width:
            raise ValueError(
                f"table[{index}]={value!r} isn't in range(0, 2**len(target)) "
                f"(len(target)={width})"
            )
        converted_rows.append(
            tuple(bool(bit) for bit in split_integer(value, width)) if width else ()
        )
    rows = tuple(converted_rows)
    if width and len(rows) != 1 << n:
        raise ValueError("Table contents must exactly fill the address space")
    trace = ComponentTrace()
    operations: tuple[ReversiblePhaseOperation, ...] = ()
    bits = classical_width
    if output:
        if erase:
            phase = tuple(range(classical_width, classical_width + (1 << n)))
            measurement = phase[-1] + 1
            operations = measured_unlookup_operations(
                rows,
                selection,
                output,
                workspace[:n],
                phase,
                measurement_bit=measurement,
                reused_phase_bits=reused_phase_bits,
                trace=trace,
            )
            bits = measurement + 1
        else:
            prefix = tuple(
                ReversiblePhaseOperation("r", (q,)) for q in reversed(output)
            )
            operations = (
                *prefix,
                *recursive_lookup_operations(
                    rows,
                    selection,
                    output,
                    workspace[:n],
                    measurement_bit=classical_width,
                    trace=trace,
                    trace_offset=len(prefix),
                ),
            )
            bits += bool(n)
    return (
        ReversiblePhaseProgram(
            max((*selection, *output, *workspace), default=0) + 1,
            bits,
            operations,
            tuple(range(classical_width)),
        ),
        tuple(trace.calls),
    )
