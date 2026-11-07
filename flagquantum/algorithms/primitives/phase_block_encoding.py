# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Phase.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/phase.py

"""Compose source scalar phases with existing native block operations."""

from dataclasses import dataclass

from ...circuit import Circuit, _CircuitOperation
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.global_phase import global_phase_ir
from ...core.ir import CircuitIR, Instruction
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare


@dataclass(frozen=True)
class PhaseBlockEncoding:
    """Apply exp(i*pi*phi) to a block with unchanged signal metadata.

    Bind the child separately using its existing operation interface. The
    scalar carrier may be an arbitrary existing system or auxiliary wire;
    its value and entanglement are preserved. No extra workspace is needed.
    """

    block_encoding: _BlockMetadata
    phi: SymbolicReal
    eps: SymbolicReal

    @property
    def system_bitsize(self) -> int:
        return self.block_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.block_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> int:
        return self.block_encoding.resource_bitsize

    @property
    def alpha(self) -> SymbolicReal:
        return self.block_encoding.alpha

    @property
    def epsilon(self) -> SymbolicReal:
        return self.block_encoding.epsilon + self.eps

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self.block_encoding.signal_state

    def resource_model(self, block_cost: ComponentCost) -> ComponentCost:
        """Retain the original scalar and block calls, independently of IR."""
        ports = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
        if block_cost.qubits < ports:
            raise ValueError("Block model must include its declared quantum ports")
        return ComponentCost(
            "PhaseBlockEncoding",
            ports,
            calls=(
                CostCall(ComponentCost("GlobalPhase", 0), bystander_qubits=ports),
                CostCall(block_cost),
            ),
        )

    def make_on(
        self, child: _CircuitOperation, *, anchor: int, adjoint: bool = False
    ) -> "_BoundPhaseBlockEncoding":
        """Bind a directional child and an existing scalar carrier for append.

        Supply a child already bound in the requested direction. Symbolic
        phase metadata is valid for models but cannot be emitted numerically.
        """
        if type(adjoint) is not bool:
            raise ValueError("Block phase direction must be boolean")
        if not isinstance(self.phi, (int, float)) or isinstance(self.phi, bool):
            raise ValueError("Executable block phase requires a concrete exponent")
        gates = global_phase_ir(-self.phi if adjoint else self.phi, anchor=anchor)
        return _BoundPhaseBlockEncoding(child, anchor, gates, adjoint)


@dataclass(frozen=True)
class _BoundPhaseBlockEncoding:
    child: _CircuitOperation
    anchor: int
    phase: tuple[Instruction, ...]
    adjoint: bool

    def append_to(self, circuit: Circuit) -> Circuit:
        """Preserve source call direction without changing the child's emitter."""
        if self.anchor >= circuit.n_qubits:
            raise ValueError("Scalar phase anchor exceeds circuit width")
        scalar = CircuitIR(
            circuit.n_qubits, self.phase, metadata={"batch_size": circuit.bsz}
        ).validate()
        if self.adjoint:
            circuit.append(self.child)
            circuit.compose(scalar)
        else:
            circuit.compose(scalar)
            circuit.append(self.child)
        return circuit
