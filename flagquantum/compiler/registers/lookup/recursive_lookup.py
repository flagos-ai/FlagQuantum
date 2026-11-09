# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_lookup, without changing its schedule.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/lookup/gen_lookup.cc

"""Source recursive lookup with quantum, classical and constant table data."""

from typing import Literal

from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..finite_field.polynomial import _validate_registers
from ..layout.workspace import RegisterWorkspace

TableBit = int | bool | ClassicalFieldCoefficient


def _lookup_leaf(
    row: tuple[TableBit, ...],
    output: tuple[int, ...],
    control: int | None,
    pauli: Literal["x", "z"],
) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for value, target in zip(row, output, strict=True):
        if value is False:
            continue
        controls = (() if control is None else (control,)) + (
            (value,) if type(value) is int else ()
        )
        bit = value.slot if isinstance(value, ClassicalFieldCoefficient) else None
        opcode = (
            ("x", "cx", "ccx")[len(controls)]
            if pauli == "x"
            else ("z", "cz", "ccz")[len(controls)]
        )
        operations.append(Operation(opcode, (*controls, target), condition_bit=bit))
    return tuple(operations)


def recursive_lookup_operations(
    table: tuple[tuple[TableBit, ...], ...],
    address: tuple[int, ...],
    output: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    pauli: Literal["x", "z"] = "x",
    control: int | None = None,
    measurement_bit: int = 0,
) -> tuple[Operation, ...]:
    """Emit the unchanged source schedule, including incomplete-table behavior.

    All registers and rows are big-endian. Missing rows do not emit a body,
    but source low-only subtrees have measurement phases documented in the
    acceptance contract. Full tables have the intended coherent lookup map.
    """
    if not output:
        return ()
    if len(address) > 30:
        raise ValueError("Portable source lookup supports at most 30 address bits")
    if (
        not table
        or len(table) > 1 << len(address)
        or any(len(row) != len(output) for row in table)
    ):
        raise ValueError(
            "Lookup needs nonempty equal-width rows fitting the address space"
        )
    if pauli not in {"x", "z"}:
        raise ValueError("Lookup pauli must be x or z")
    if type(measurement_bit) is not int or measurement_bit < 0:
        raise ValueError("Lookup erasure requires a nonnegative result slot")
    values = tuple(value for row in table for value in row)
    if any(
        type(value) not in {int, bool}
        and not isinstance(value, ClassicalFieldCoefficient)
        for value in values
    ):
        raise ValueError(
            "Lookup entries must be quantum wires, classical slots or booleans"
        )
    quantum = tuple(sorted({value for value in values if type(value) is int}))
    _validate_registers(
        address, output, workspace, () if control is None else (control,), quantum
    )
    if address and any(
        isinstance(value, ClassicalFieldCoefficient) and value.slot == measurement_bit
        for value in values
    ):
        raise ValueError("Lookup result slot cannot overwrite a classical table input")
    pool = RegisterWorkspace(workspace)
    operations: list[Operation] = []
    source_rows = tuple(row[::-1] for row in table)
    source_output = output[::-1]

    def emit(
        rows: tuple[tuple[TableBit, ...], ...],
        word: tuple[int, ...],
        parent: int | None,
    ) -> None:
        if not word:
            operations.extend(_lookup_leaf(rows[0], source_output, parent, pauli))
            return
        lease = pool.allocate(1)
        anc = lease.wires[0]
        bit = word[-1]
        controls = () if parent is None else (parent,)
        operations.extend(
            (
                Operation("x", (bit,)),
                Operation("r", (anc,)),
                Operation("cx" if parent is None else "ccx", (*controls, bit, anc)),
                Operation("x", (bit,)),
            )
        )
        half = 1 << (len(word) - 1)
        if len(rows) > half:
            emit(rows[:half], word[:-1], anc)
            operations.append(
                Operation("x" if parent is None else "cx", (*controls, anc))
            )
            emit(rows[half:], word[:-1], anc)
        else:
            emit(rows, word[:-1], anc)
        operations.extend(
            (
                Operation("hmr", (anc,), result_bit=measurement_bit),
                Operation(
                    "z" if parent is None else "cz",
                    (*controls, bit),
                    condition_bit=measurement_bit,
                ),
            )
        )
        pool.free(lease)

    emit(source_rows, address[::-1], control)
    return tuple(operations)
