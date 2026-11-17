# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BlackBoxPrepare and its AutoPartition binding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/black_box_prepare.py

"""Declare packed PREPARE ports and bind concrete preparation emitters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.auto_partition import AutoPartition
from ...compiler.registers.layout.partition import RegisterLayout
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import opaque_register_type
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
        """Bind a native emitter once the named layout and its norm are concrete."""
        selection = _concrete_layout(self.prepare.selection_registers)
        junk = _concrete_layout(self.prepare.junk_registers)
        try:
            normalization = self.l1_norm_of_coeffs
        except NotImplementedError:
            normalization = None
        if normalization is not None and not isinstance(normalization, (int, float)):
            raise ValueError("PREPARE binding requires a concrete normalization")
        return BlackBoxPrepare(selection, junk, emit, normalization)

    def __str__(self) -> str:
        return "Prep"


def _concrete_layout(registers: tuple[Register, ...]) -> RegisterLayout:
    layout: list[tuple[str, int]] = []
    for reg in registers:
        width = reg.total_bits()
        if not isinstance(width, int):
            raise ValueError(
                f"PREPARE binding requires a concrete width for {reg.name}"
            )
        layout.append((reg.name, width))
    return tuple(layout) if any(width for _, width in layout) else ()


@dataclass(frozen=True)
class BlackBoxPrepare:
    """Bind a preparation emitter while preserving the order of its two port groups."""

    selection_layout: RegisterLayout
    junk_layout: RegisterLayout
    prepare: PrepareEmitter
    normalization: float | None = None

    @property
    def selection_width(self) -> int:
        return sum(width for _, width in self.selection_layout)

    @property
    def junk_width(self) -> int:
        return sum(width for _, width in self.junk_layout)

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
        ports, _ = AutoPartition(
            (("selection", self.selection_layout), ("junk", self.junk_layout))
        ).bind({"selection": selection, "junk": junk})
        allowed = (*selection, *junk, *workspace)
        if allowed:
            _validate_wires(allowed)
        if not selection:
            return ()
        gates = self.prepare(ports, workspace, classical_bits, adjoint)
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
