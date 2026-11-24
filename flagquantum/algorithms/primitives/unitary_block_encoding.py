# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Unitary.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/unitary.py

"""Bind a THRU unitary as a block encoding with untouched auxiliary ports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.control.control_predicate import (
    ControlRegister,
    bind_control_registers,
)
from ...compiler.registers.control.controlled_component import controlled_component_ir
from ...compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from ...compiler.registers.layout.packed_block_component import unitary_block_signature
from ...compiler.registers.layout.partition import (
    RegisterLayout,
    partition_register,
    partition_width,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _validate_block_work
from ._block_metadata import _packed_block_signature
from .black_box_prepare import BlackBoxPrepare, _identity_prepare

_UnitaryEmitter = Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    tuple[Instruction, ...],
]


_ControlledUnitaryEmitter = Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    tuple[Instruction, ...],
]


def source_unitary_block_signature(
    child: Signature, *, ancilla: SymbolicInt = 0, resource: SymbolicInt = 0
) -> Signature:
    """Declare source widths, which sum the child element widths independently of shape."""
    return unitary_block_signature(child, ancilla=ancilla, resource=resource)


def source_unitary_block_resource_formula(
    child: SymbolicResourceFormula,
    *,
    ancilla: SymbolicInt = 0,
    resource: SymbolicInt = 0,
    system: SymbolicInt | None = None,
) -> SymbolicResourceFormula:
    """Track the source partition, child call and idle auxiliary ports."""
    system_width = child.logical_qubits if system is None else system
    auxiliary = ancilla + resource
    partition_peak = cast(SymbolicInt, smax(0, system_width))
    peak = cast(
        SymbolicInt,
        smax(
            0,
            system_width + auxiliary,
            partition_peak + auxiliary,
            child.logical_qubits + auxiliary,
        ),
    )
    return replace(child, logical_qubits=peak)


@dataclass(frozen=True)
class UnitaryBlockEncoding:
    """Wrap a register-preserving child without changing its decomposition.

    System ports use declared big-endian layout order. Ancilla/resource ports
    pass through untouched, even when not zero. Child workspace remains owned
    by the caller and must be cleaned by the child in either direction.
    """

    system_layout: RegisterLayout
    unitary: _UnitaryEmitter
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

    @classmethod
    def from_signature(
        cls,
        signature: Signature,
        unitary: _UnitaryEmitter,
        *,
        alpha: SymbolicReal = 1,
        ancilla_bitsize: int = 0,
        resource_bitsize: int = 0,
        epsilon: SymbolicReal = 0,
    ) -> "UnitaryBlockEncoding":
        """Preserve typed child port order for concrete quantum registers."""
        if not signature.thru_registers_only:
            raise ValueError("Block encoded unitary must have all THRU registers")
        layout: list[tuple[str, int]] = []
        for register in signature:
            width = register.dtype.num_qubits
            if (
                any(type(dim) is not int for dim in register.shape_symbolic)
                or register.dtype.num_cbits != 0
                or type(width) is not int
            ):
                raise ValueError("Unitary binding requires concrete quantum registers")
            if register.total_bits() != width:
                raise ValueError(
                    "Unitary source packed width does not match child shape"
                )
            layout.append((register.name, width))
        return cls(
            tuple(layout),
            unitary,
            alpha=alpha,
            ancilla_bitsize=ancilla_bitsize,
            resource_bitsize=resource_bitsize,
            epsilon=epsilon,
        )

    @property
    def system_bitsize(self) -> int:
        return sum(width for _, width in self.system_layout)

    @property
    def signature(self) -> Signature:
        """Return the packed source system, ancilla and resource declaration."""
        return _packed_block_signature(
            self.system_bitsize, self.ancilla_bitsize, self.resource_bitsize
        )

    @property
    def signal_state(self) -> BlackBoxPrepare:
        """Expose the source identity preparation on the ancilla register."""
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(self, unitary_cost: ComponentCost) -> ComponentCost:
        """Charge one directional child plus all idle auxiliary bystanders."""
        return self._child_resource_model(unitary_cost, control_width=0)

    def _child_resource_model(
        self, unitary_cost: ComponentCost, *, control_width: int
    ) -> ComponentCost:
        width = self.system_bitsize + control_width
        if unitary_cost.qubits < width:
            raise ValueError("Child model must include every system and control port")
        auxiliary = self.ancilla_bitsize + self.resource_bitsize
        return ComponentCost(
            "UnitaryBlockEncoding",
            width + auxiliary,
            calls=(CostCall(unitary_cost, bystander_qubits=auxiliary),),
        )

    def controlled_resource_model(
        self,
        positive_child: ComponentCost,
        registers: tuple[ControlRegister, ...],
        *,
        legacy_shims: bool = False,
    ) -> ComponentCost:
        """Wrap the source positive child cost with optional predicate work."""
        positive = self._child_resource_model(positive_child, control_width=1)
        return controlled_component_cost(
            positive,
            registers,
            target_width=positive.qubits - 1,
            legacy_shims=legacy_shims,
        )

    def make_controlled_on(
        self,
        emit_child: _ControlledUnitaryEmitter,
        *,
        controls: tuple[tuple[int, ...], ...],
        control_registers: tuple[ControlRegister, ...],
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        control_workspace: tuple[int, ...] = (),
        control_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind the directional positive child under the shared control predicate."""
        self._validate_ports(system, ancilla, resource, adjoint)
        control_wires, _ = bind_control_registers(controls, control_registers)
        existing = (*control_wires, *system, *ancilla, *resource, *workspace)
        _validate_block_work(
            existing, control_workspace, (*classical_bits, *control_classical_bits)
        )
        ports = partition_register(system, self.system_layout)

        def emit(predicate: int) -> tuple[Instruction, ...]:
            gates = emit_child(predicate, ports, workspace, classical_bits, adjoint)
            _check_child_wires(gates, {predicate, *system, *workspace})
            return gates

        gates = controlled_component_ir(
            controls, control_registers, emit, control_workspace, control_classical_bits
        )
        return _BoundBlockOperation(gates, max((*existing, *control_workspace)) + 1)

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
        self._validate_ports(system, ancilla, resource, adjoint)
        _validate_block_work((*system, *ancilla, *resource), workspace, classical_bits)
        return _BoundUnitaryBlockEncoding(
            self, system, ancilla, resource, workspace, classical_bits, adjoint
        )

    def _validate_ports(
        self,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        resource: tuple[int, ...],
        adjoint: bool,
    ) -> None:
        if type(adjoint) is not bool:
            raise ValueError("Block direction must be boolean")
        partition_register(system, self.system_layout)
        if (
            len(ancilla) != self.ancilla_bitsize
            or len(resource) != self.resource_bitsize
        ):
            raise ValueError("Block auxiliary ports must match their declared widths")


def _check_child_wires(gates: tuple[Instruction, ...], allowed: set[int]) -> None:
    if any(wire not in allowed for gate in gates for wire in gate.wires):
        raise ValueError("Block child used a wire outside its system and workspace")


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
        _check_child_wires(gates, {*self.system, *self.workspace})
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
