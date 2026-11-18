# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 multiplexers/black_box_select.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/black_box_select.py

"""Resolve packed SELECT declarations into the existing named SELECT consumer."""

from dataclasses import dataclass
from functools import cached_property

from ...compiler.registers.layout.select_partition import packed_select_layout
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, ssum
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .select_block_encoding import SelectBlockEncoding, SelectEmitter
from .select_oracle import SelectOracle


@dataclass(frozen=True)
class PackedSelectDeclaration:
    """Retain the original packed metadata and bind concrete through-port emitters."""

    select: SelectOracle

    @cached_property
    def selection_bitsize(self) -> SymbolicInt:
        return ssum(reg.total_bits() for reg in self.select.selection_registers)

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return ssum(reg.total_bits() for reg in self.select.target_registers)

    @cached_property
    def control_registers(self) -> tuple[Register, ...]:
        return self.select.control_registers

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("selection", opaque_register_type(self.selection_bitsize)),)

    @cached_property
    def target_registers(self) -> tuple[Register, ...]:
        return (Register("system", opaque_register_type(self.system_bitsize)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature((*self.selection_registers, *self.target_registers))

    def bind(self, emit: SelectEmitter) -> SelectBlockEncoding:
        """Bind an uncontrolled quantum THRU child on caller-owned work and slots."""
        if self.control_registers:
            raise ValueError("Packed SELECT requires an uncontrolled child")
        native_groups = []
        for registers in (
            self.select.selection_registers,
            self.select.target_registers,
        ):
            group = []
            for register in registers:
                if register.side.name != "THRU" or register.total_cbits() != 0:
                    raise ValueError("Packed SELECT requires quantum THRU registers")
                width = register.total_bits()
                if not isinstance(width, int):
                    raise ValueError(
                        f"SELECT binding requires a concrete width for {register.name}"
                    )
                group.append(Register(register.name, opaque_register_type(width)))
            native_groups.append(tuple(group))
        selection_registers, target_registers = native_groups
        signature, _ = packed_select_layout(selection_registers, target_registers)
        assert signature == self.signature
        layouts = []
        for registers in (selection_registers, target_registers):
            layout = []
            for register in registers:
                width = register.total_bits()
                assert isinstance(width, int)
                layout.append((register.name, width))
            layouts.append(tuple(layout))
        selection, target = layouts
        return SelectBlockEncoding(
            selection,
            target,
            emit,
            BlackBoxPrepare(selection, (), _identity_prepare),
        )

    def __str__(self) -> str:
        return "SELECT"
