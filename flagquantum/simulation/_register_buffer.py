# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Sim register buffer transposition.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Transpose LSB-first scalar register rows and existing packed basis wires."""

from collections.abc import Sequence
from typing import Protocol

from ..core._fixed_width_integer import FixedWidthInteger
from .physical_basis_shots import PackedBasisState


class _Register(Protocol):
    @property
    def members(self) -> tuple[tuple[str, int], ...]: ...


def pack_register_buffer(
    registers: Sequence[_Register],
    rows: Sequence[Sequence[FixedWidthInteger]],
    state: PackedBasisState,
) -> None:
    """Assign declared wires in register order, including overlapping members."""
    _validate_buffer(registers, rows, state)
    for column, register in enumerate(registers):
        for offset, (kind, wire) in enumerate(register.members):
            target = state.qubits if kind == "q" else state.bits
            target[wire] = sum(
                int(row[column].bit(offset)) << lane for lane, row in enumerate(rows)
            )


def unpack_register_buffer(
    registers: Sequence[_Register],
    rows: list[list[FixedWidthInteger]],
    state: PackedBasisState,
) -> None:
    """Replace scalar cells while preserving row ownership and clearing hanging bits."""
    _validate_buffer(registers, rows, state)
    for lane, row in enumerate(rows):
        for column, register in enumerate(registers):
            value = 0
            for offset, (kind, wire) in enumerate(register.members):
                source = state.qubits if kind == "q" else state.bits
                value |= ((source[wire] >> lane) & 1) << offset
            row[column] = FixedWidthInteger(len(register.members), value)


def _validate_buffer(
    registers: Sequence[_Register],
    rows: Sequence[Sequence[FixedWidthInteger]],
    state: PackedBasisState,
) -> None:
    if len(rows) != len(state.phases):
        raise ValueError("Register rows must match the packed lane count")
    for row in rows:
        if len(row) != len(registers) or any(
            cell.width != len(register.members)
            for cell, register in zip(row, registers, strict=True)
        ):
            raise ValueError("Register cells must match the declared layout")
