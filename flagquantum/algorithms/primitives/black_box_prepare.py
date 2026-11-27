# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BlackBoxPrepare and its AutoPartition binding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/black_box_prepare.py

"""Declare packed PREPARE ports and bind concrete preparation emitters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from math import prod

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.auto_partition import AutoPartition
from ...compiler.registers.layout.partition import RegisterLayout
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.register_verifier import executable_register_width
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal, ssum
from .prepare_oracle import PrepareOracle

PrepareEmitter = Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    tuple[Instruction, ...],
]


def _identity_prepare(
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return ()


@dataclass(frozen=True)
class PackedPrepareDeclaration:
    """Preserve the packed metadata of a native PREPARE oracle."""

    prepare: PrepareOracle

    @cached_property
    def selection_width(self) -> SymbolicInt:
        return ssum(reg.total_bits() for reg in self.prepare.selection_registers)

    @cached_property
    def junk_width(self) -> SymbolicInt:
        return ssum(reg.total_bits() for reg in self.prepare.junk_registers)

    @cached_property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("selection", opaque_register_type(self.selection_width)),)

    @cached_property
    def junk_registers(self) -> tuple[Register, ...]:
        return (Register("junk", opaque_register_type(self.junk_width)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(selection=self.selection_width, junk=self.junk_width)

    @cached_property
    def l1_norm_of_coeffs(self) -> SymbolicReal:
        return self.prepare.l1_norm_of_coeffs

    def bind(self, emit: PrepareEmitter) -> BlackBoxPrepare:
        """Bind concrete named ports while preserving a scalar normalization expression."""
        selection = _concrete_layout(self.prepare.selection_registers)
        junk = _concrete_layout(self.prepare.junk_registers)
        try:
            normalization = self.l1_norm_of_coeffs
        except NotImplementedError:
            normalization = None
        if normalization is not None and not isinstance(normalization, (int, float)):
            from sympy import Expr

            if not isinstance(normalization, Expr):
                raise ValueError(
                    "PREPARE normalization must be a native scalar expression"
                )
        return BlackBoxPrepare(selection, junk, emit, normalization)

    def __str__(self) -> str:
        return "Prep"


def _concrete_layout(registers: tuple[Register, ...]) -> RegisterLayout:
    layout: list[tuple[str, int]] = []
    for reg in registers:
        try:
            width = executable_register_width(reg.dtype) * prod(reg.shape)
        except ValueError as error:
            raise ValueError(
                f"PREPARE binding requires a concrete width for {reg.name}"
            ) from error
        layout.append((reg.name, width))
    return tuple(layout) if any(width for _, width in layout) else ()


@dataclass(frozen=True)
class BlackBoxPrepare:
    """Bind logical ports and optional preparation-local output storage.

    Local junk occupies the workspace prefix. It starts clean before PREPARE,
    stays live across SELECT, and is consumed by PREPARE adjoint. It is not
    exposed as a logical through-junk operand. Remaining workspace is scratch.
    """

    selection_layout: RegisterLayout
    junk_layout: RegisterLayout
    prepare: PrepareEmitter
    normalization: SymbolicReal | None = None
    local_junk_layout: RegisterLayout = ()

    @property
    def selection_width(self) -> int:
        return sum(width for _, width in self.selection_layout)

    @property
    def junk_width(self) -> int:
        return sum(width for _, width in self.junk_layout)

    @property
    def local_junk_width(self) -> int:
        return sum(width for _, width in self.local_junk_layout)

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        """Return the opaque packed selection descriptor, including width zero."""
        return (Register("selection", opaque_register_type(self.selection_width)),)

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        """Return the opaque packed junk descriptor, including width zero."""
        return (Register("junk", opaque_register_type(self.junk_width)),)

    @property
    def signature(self) -> Signature:
        """Describe the original scalar packed through ports."""
        return Signature.build(selection=self.selection_width, junk=self.junk_width)

    def make_on(
        self,
        *,
        selection: tuple[int, ...],
        junk: tuple[int, ...],
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> "_BoundBlackBoxPrepare":
        """Bind the packed ports and explicit scratch for circuit.append."""
        return _BoundBlackBoxPrepare(
            self, selection, junk, workspace, classical_bits, adjoint
        )

    def emit(
        self,
        selection: tuple[int, ...],
        junk: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        """Lower the child on the same physical qubits with no partition gates."""
        if type(adjoint) is not bool:
            raise ValueError("Preparation direction must be boolean")
        if {name for name, _ in self.selection_layout} & {
            name for name, _ in self.junk_layout
        }:
            raise ValueError("Preparation selection and junk names must be distinct")
        local_width = self.local_junk_width
        if len(workspace) < local_width:
            raise ValueError("Preparation needs its persistent local junk storage")
        ports, _ = AutoPartition(
            (
                ("selection", self.selection_layout),
                ("junk", self.junk_layout),
                ("local", self.local_junk_layout),
            )
        ).bind(
            {
                "selection": selection,
                "junk": junk,
                "local": workspace[:local_width],
            }
        )
        allowed = (*selection, *junk, *workspace)
        if allowed:
            _validate_wires(allowed)
        if not selection and not local_width:
            return ()
        gates = self.prepare(ports, workspace[local_width:], classical_bits, adjoint)
        qubits = set(allowed)
        if any(q not in qubits for gate in gates for q in gate.wires):
            raise ValueError("Preparation used a qubit outside the bound registers")
        return gates


@dataclass(frozen=True)
class _BoundBlackBoxPrepare:
    prepare: BlackBoxPrepare
    selection: tuple[int, ...]
    junk: tuple[int, ...]
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]
    adjoint: bool

    def append_to(self, circuit: Circuit) -> Circuit:
        if any(
            q >= circuit.n_qubits
            for q in (*self.selection, *self.junk, *self.workspace)
        ):
            raise ValueError("PREPARE binding exceeds circuit width")
        gates = self.prepare.emit(
            self.selection,
            self.junk,
            self.workspace,
            self.classical_bits,
            adjoint=self.adjoint,
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
