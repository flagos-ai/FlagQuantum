# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectBlockEncoding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/lcu_block_encoding.py

"""A block encoding with a separate prepared signal state."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.partition import RegisterLayout, partition_register
from ...core.ir import CircuitIR, Instruction
from .black_box_prepare import BlackBoxPrepare

SelectEmitter = Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    tuple[Instruction, ...],
]


@dataclass(frozen=True)
class SelectBlockEncoding:
    """Apply SELECT and expose PREPARE separately as the signal state.

    SELECT's big-endian selection and target layouts describe the execution
    ports. PREPARE's junk belongs to the signal state and has no SELECT port.
    """

    selection_layout: RegisterLayout
    target_layout: RegisterLayout
    select: SelectEmitter
    signal_state: BlackBoxPrepare

    @property
    def ancilla_bitsize(self) -> int:
        return self.signal_state.selection_width

    @property
    def resource_bitsize(self) -> int:
        return self.signal_state.junk_width

    @property
    def system_bitsize(self) -> int:
        return sum(width for _, width in self.target_layout)

    @property
    def alpha(self) -> float:
        normalization = self.signal_state.normalization
        if normalization is None:
            raise ValueError("The signal state must declare its L1 normalization")
        return normalization

    @property
    def epsilon(self) -> float:
        """Return the pinned source placeholder for approximation error."""
        return 0.0

    def resource_model(self, select_cost: ComponentCost) -> ComponentCost:
        """Wrap one SELECT cost without charging the signal preparation."""
        selection_width = sum(width for _, width in self.selection_layout)
        return ComponentCost(
            "SelectBlockEncoding",
            selection_width + self.system_bitsize,
            calls=(CostCall(select_cost),),
        )

    def make_on(
        self,
        *,
        selection: tuple[int, ...],
        target: tuple[int, ...],
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> "_BoundSelectBlockEncoding":
        """Bind SELECT operands and caller-owned clean work for circuit.append."""
        return _BoundSelectBlockEncoding(
            self, selection, target, workspace, classical_bits, adjoint
        )

    def emit(
        self,
        selection: tuple[int, ...],
        target: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        *,
        adjoint: bool = False,
    ) -> tuple[Instruction, ...]:
        """Invoke the SELECT emitter on its named ports."""
        if type(adjoint) is not bool:
            raise ValueError("SELECT direction must be boolean")
        if {name for name, _ in self.selection_layout} & {
            name for name, _ in self.target_layout
        }:
            raise ValueError("SELECT selection and target port names must be distinct")
        ports = partition_register(selection, self.selection_layout)
        ports.update(partition_register(target, self.target_layout))
        allowed = (*selection, *target, *workspace)
        if allowed:
            _validate_wires(allowed)
        gates = self.select(ports, workspace, classical_bits, adjoint)
        qubits = set(allowed)
        if any(q not in qubits for gate in gates for q in gate.wires):
            raise ValueError("SELECT used a qubit outside the bound registers")
        return gates


@dataclass(frozen=True)
class _BoundSelectBlockEncoding:
    encoding: SelectBlockEncoding
    selection: tuple[int, ...]
    target: tuple[int, ...]
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]
    adjoint: bool

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append SELECT to the circuit in the bound register order."""
        gates = self.encoding.emit(
            self.selection,
            self.target,
            self.workspace,
            self.classical_bits,
            adjoint=self.adjoint,
        )
        if any(
            wire >= circuit.n_qubits
            for wire in (*self.selection, *self.target, *self.workspace)
        ):
            raise ValueError("SELECT binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
