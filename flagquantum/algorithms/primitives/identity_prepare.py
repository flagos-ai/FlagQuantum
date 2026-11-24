# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 reflections/prepare_identity.py.

"""Identity signal preparation through the existing named PREPARE boundary."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ...compiler.registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from ...compiler.registers.primitives.identity import register_identity_ir
from ...core.ir import Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.value_math import SymbolicInt
from .black_box_prepare import BlackBoxPrepare, PackedPrepareDeclaration
from .prepare_oracle import PrepareOracle


@dataclass(frozen=True, init=False)
class IdentityPrepare(PrepareOracle):
    """Declare identity selection ports without assuming a coefficient norm."""

    selection_regs: tuple[Register, ...]

    def __init__(self, selection_regs: Register | Iterable[Register]) -> None:
        object.__setattr__(
            self,
            "selection_regs",
            (
                (selection_regs,)
                if isinstance(selection_regs, Register)
                else tuple(selection_regs)
            ),
        )

    @classmethod
    def from_bitsizes(cls, bitsizes: Sequence[SymbolicInt]) -> "IdentityPrepare":
        """Keep original generated names and opaque register widths."""
        return cls(
            Register(f"reg{i}_", opaque_register_type(width))
            for i, width in enumerate(bitsizes)
        )

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return self.selection_regs

    def adjoint(self) -> "IdentityPrepare":
        return self

    def bind(self) -> BlackBoxPrepare:
        """Bind concrete scalar through ports to the existing append operation.

        Symbolic widths remain declaration-only. Array and one-sided source
        ports lack an executable Identity body and fail before physical binding.
        """
        if any(
            reg.shape_symbolic or reg.side != RegisterSide.THRU
            for reg in self.selection_regs
        ):
            raise ValueError("Identity preparation requires scalar through ports")
        if any(reg.bitsize == 0 for reg in self.selection_regs):
            raise ValueError("Identity preparation source has no zero-width body")
        if any(reg.dtype.num_cbits != 0 for reg in self.selection_regs):
            raise ValueError("Identity preparation requires quantum selection ports")
        return PackedPrepareDeclaration(self).bind(_emit_identity)


def _emit_identity(
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return tuple(
        gate for wires in ports.values() for gate in register_identity_ir(wires)
    )


def identity_prepare_program(prepare: IdentityPrepare) -> ProgramRecord:
    """Retain one atomic word Identity call per concrete selection port.

    Words are MSB-first THRU storage. No work or measurement is needed. The
    original source lacks executable shaped, one-sided and zero-width bodies.
    """

    prepare.bind()
    return identity_preparation_program(prepare.selection_registers)
