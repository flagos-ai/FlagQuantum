# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_unlookup, preserving the source call chain.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/lookup/gen_lookup.cc

"""Measured table erasure through shared classical parity and unary lookup."""

from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ..bits.binary_unary_conversion import (
    binary_to_unary_operations,
    unary_to_binary_operations,
)
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..layout.workspace import RegisterWorkspace
from .recursive_lookup import recursive_lookup_operations


def measured_unlookup_operations(
    table: tuple[tuple[bool | ClassicalFieldCoefficient, ...], ...],
    address: tuple[int, ...],
    output: tuple[int, ...],
    workspace: tuple[int, ...],
    phase_bits: tuple[int, ...],
    *,
    measurement_bit: int,
    reused_phase_bits: tuple[int, ...] = (),
) -> tuple[Operation, ...]:
    """Erase computed table output, preserving source allocation and gate order.

    Quantum registers and table rows are big-endian; phase slots are indexed
    by integer address. Workspace is clean and consumed in caller order.
    A reused phase slot emits the source allocator's extra initialization.
    """
    if not output:
        return ()
    _validate_registers(address, output, workspace)
    n = len(address)
    if n > 30:
        raise ValueError("Portable source unlookup supports at most 30 address bits")
    if len(table) > 1 << n or any(len(row) != len(output) for row in table):
        raise ValueError("Unlookup rows must match output width and fit the address")
    values = tuple(value for row in table for value in row)
    if any(
        type(value) is not bool and not isinstance(value, ClassicalFieldCoefficient)
        for value in values
    ):
        raise ValueError(
            "Unlookup rejects quantum data; table entries must be bool or classical slots"
        )
    slots = (*phase_bits, measurement_bit)
    if (
        len(phase_bits) != 1 << n
        or len(set(slots)) != len(slots)
        or any(type(slot) is not int or slot < 0 for slot in slots)
    ):
        raise ValueError(
            "Unlookup requires distinct phase slots and a separate measurement slot"
        )
    inputs = {
        value.slot for value in values if isinstance(value, ClassicalFieldCoefficient)
    }
    if inputs.intersection(slots):
        raise ValueError("Unlookup classical work cannot overwrite table inputs")
    if reused_phase_bits != phase_bits[: len(reused_phase_bits)] or any(
        type(slot) is not int for slot in reused_phase_bits
    ):
        raise ValueError("Reused phase slots must be an ordered prefix of phase_bits")
    operations = [
        Operation("bit_store0", (), result_bit=slot)
        for slot in phase_bits
        if slot in reused_phase_bits
    ]
    operations.extend(
        Operation("bit_store0", (), result_bit=slot) for slot in phase_bits
    )
    rows = tuple(row[::-1] for row in table)
    for j, wire in enumerate(reversed(output)):
        operations.extend(
            (
                Operation("hmr", (wire,), result_bit=measurement_bit),
                Operation("push_condition", (), condition_bit=measurement_bit),
            )
        )
        for row_index, row in enumerate(rows):
            value = row[j]
            if value is False:
                continue
            operations.append(
                Operation(
                    "bit_invert",
                    (),
                    result_bit=phase_bits[row_index],
                    condition_bit=(
                        value.slot
                        if isinstance(value, ClassicalFieldCoefficient)
                        else None
                    ),
                )
            )
        operations.append(Operation("pop_condition", ()))
    h = n // 2
    size = 1 << h
    source_address = address[::-1]
    unfolded = [*source_address[:h], *output[::-1][: size - h]]
    pool = RegisterWorkspace(workspace)
    extra = max(0, size - len(unfolded))
    lease = pool.allocate(extra)
    unfolded.extend(lease.wires)
    buffer = tuple(reversed(unfolded))
    operations.extend(binary_to_unary_operations(buffer))
    phase_rows = tuple(
        tuple(
            ClassicalFieldCoefficient(slot)
            for slot in reversed(phase_bits[start : start + size])
        )
        for start in range(0, len(phase_bits), size)
    )
    operations.extend(
        recursive_lookup_operations(
            phase_rows,
            tuple(reversed(source_address[h:])),
            buffer,
            workspace[extra:],
            pauli="z",
            measurement_bit=measurement_bit,
        )
    )
    operations.extend(
        unary_to_binary_operations(buffer, measurement_bit=measurement_bit)
    )
    pool.free(lease)
    return tuple(operations)
