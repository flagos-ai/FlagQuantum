# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Unitary.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/unitary.py

"""Bind a THRU unitary as a block encoding with untouched auxiliary ports."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import (
    RegisterLayout,
    partition_register,
    partition_width,
)
from ...core.ir import CircuitIR, Instruction
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


@dataclass(frozen=True)
class UnitaryBlockEncoding:
    """Wrap a register-preserving child without changing its decomposition.

    System ports use declared big-endian layout order. Ancilla/resource ports
    pass through untouched, even when not zero. Child workspace remains owned
    by the caller and must be cleaned by the child in either direction.
    """

    system_layout: RegisterLayout
    unitary: Callable[
        [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        tuple[Instruction, ...],
    ]
    alpha: SymbolicReal = 1
    ancilla_bitsize: int = 0
    resource_bitsize: int = 0
    epsilon: SymbolicReal = 0

    def __post_init__(self) -> None:
        # Validate the existing layout contract without a second register type.
        if any(type(width) is not int for _, width in self.system_layout):
            raise ValueError("Unitary system layout requires concrete integer widths")
        partition_width(self.system_layout)
        if any(
            type(width) is not int or width < 0
            for width in (self.ancilla_bitsize, self.resource_bitsize)
        ):
            raise ValueError("Block auxiliary widths must be nonnegative integers")

    @property
    def system_bitsize(self) -> int:
        return sum(width for _, width in self.system_layout)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        """Expose the source identity preparation on the ancilla register."""
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(self, unitary_cost: ComponentCost) -> ComponentCost:
        """Charge one directional child plus all idle auxiliary bystanders."""
        if unitary_cost.qubits < self.system_bitsize:
            raise ValueError("Child model must include every system port")
        auxiliary = self.ancilla_bitsize + self.resource_bitsize
        return ComponentCost(
            "UnitaryBlockEncoding",
            self.system_bitsize + auxiliary,
            calls=(CostCall(unitary_cost, bystander_qubits=auxiliary),),
        )

    def make_on(
        self,
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> "_BoundUnitaryBlockEncoding":
        """Bind the source packed signature for circuit.append(op)."""
        if type(adjoint) is not bool:
            raise ValueError("Block direction must be boolean")
        partition_register(system, self.system_layout)
        if (
            len(ancilla) != self.ancilla_bitsize
            or len(resource) != self.resource_bitsize
        ):
            raise ValueError("Block auxiliary ports must match their declared widths")
        wires = (*system, *ancilla, *resource, *workspace)
        if wires:
            _validate_wires(wires)
        if len(set(classical_bits)) != len(classical_bits) or any(
            type(bit) is not int or bit < 0 for bit in classical_bits
        ):
            raise ValueError("Child result slots must be distinct nonnegative integers")
        return _BoundUnitaryBlockEncoding(
            self, system, ancilla, resource, workspace, classical_bits, adjoint
        )


@dataclass(frozen=True)
class _BoundUnitaryBlockEncoding:
    encoding: UnitaryBlockEncoding
    system: tuple[int, ...]
    ancilla: tuple[int, ...]
    resource: tuple[int, ...]
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]
    adjoint: bool

    def to_ir(self) -> tuple[Instruction, ...]:
        """Lower the child on its original ports; no partition gates are emitted."""
        ports = partition_register(self.system, self.encoding.system_layout)
        gates = self.encoding.unitary(
            ports, self.workspace, self.classical_bits, self.adjoint
        )
        allowed = set((*self.system, *self.workspace))
        if any(wire not in allowed for gate in gates for wire in gate.wires):
            raise ValueError("Block child used a wire outside its system and workspace")
        return gates

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append through the existing CircuitIR composition boundary."""
        if any(
            wire >= circuit.n_qubits
            for wire in (*self.system, *self.ancilla, *self.resource, *self.workspace)
        ):
            raise ValueError("Block binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, self.to_ir(), metadata={"batch_size": circuit.bsz}
            ).validate()
        )
