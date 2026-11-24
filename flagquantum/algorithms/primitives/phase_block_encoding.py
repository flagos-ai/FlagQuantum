# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Phase.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/phase.py

"""Compose source scalar phases with existing native block operations."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

from ...circuit import Circuit, _CircuitOperation
from ...compiler._component_trace.compound_control_resource import global_phase_cost_key
from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.control.control_predicate import (
    ControlRegister,
    bind_control_registers,
)
from ...compiler.registers.control.controlled_component import controlled_component_ir
from ...compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from ...compiler.registers.primitives.global_phase import global_phase_ir
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata, _packed_block_signature
from .black_box_prepare import BlackBoxPrepare


def source_phase_block_signature(
    system: SymbolicInt, ancilla: SymbolicInt = 0, resource: SymbolicInt = 0
) -> Signature:
    """Declare the locked packed ports without binding symbolic child wires."""
    return _packed_block_signature(system, ancilla, resource)


def source_phase_block_resource_formula(
    signature: Signature, child: SymbolicResourceFormula
) -> SymbolicResourceFormula:
    """Charge no QEC gates for the wire-free source scalar."""
    peak = cast(SymbolicInt, smax(signature.n_qubits(), child.logical_qubits))
    return replace(child, logical_qubits=peak)


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
    def signature(self) -> Signature:
        return source_phase_block_signature(
            self.system_bitsize, self.ancilla_bitsize, self.resource_bitsize
        )

    @property
    def alpha(self) -> SymbolicReal:
        return self.block_encoding.alpha

    @property
    def epsilon(self) -> SymbolicReal:
        return self.block_encoding.epsilon + self.eps

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self.block_encoding.signal_state

    def label(self, child_label: str) -> str:
        """Format the block annotation with an explicit native child symbol.

        The exponent is in pi units, matching the frozen source caption.
        Supplying the child symbol avoids depending on an SDK object's repr.
        """
        return f"B[exp({self.phi}i){child_label}]"

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

    def controlled_resource_model(
        self,
        positive_child: ComponentCost,
        registers: tuple[ControlRegister, ...],
        *,
        adjoint: bool = False,
        legacy_shims: bool = False,
    ) -> ComponentCost:
        """Model source Z power, controlled child and any enclosing predicate."""
        if type(adjoint) is not bool:
            raise ValueError("Block phase direction must be boolean")
        ports = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
        if positive_child.qubits < ports + 1:
            raise ValueError(
                "Positive child cost must include its control and through ports"
            )
        exponent = -self.phi if adjoint else self.phi
        family = classify_resource_gate(global_phase_cost_key(exponent, (1,), self.eps))
        scalar = CostCall(
            ComponentCost(
                "ZPowGate",
                1,
                GateCost(
                    t=int(family == "t"),
                    clifford=int(family == "clifford"),
                    rotation=int(family == "rotation"),
                ),
            ),
            bystander_qubits=ports,
        )
        child = CostCall(positive_child)
        calls = (child, scalar)
        return controlled_component_cost(
            ComponentCost("PhaseBlockEncoding", ports + 1, calls=calls),
            registers,
            target_width=ports,
            legacy_shims=legacy_shims,
        )

    def _exponent(self, adjoint: bool) -> float:
        if type(adjoint) is not bool:
            raise ValueError("Block phase direction must be boolean")
        if not isinstance(self.phi, (int, float)) or isinstance(self.phi, bool):
            raise ValueError("Executable block phase requires a concrete exponent")
        return -self.phi if adjoint else self.phi

    def make_controlled_on(
        self,
        bind_child: Callable[[int, bool], _CircuitOperation],
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
        """Bind one controlled child and a scalar on the same live predicate.

        The callback supplies a child in the requested direction, using the
        declared child work and classical slots. It preserves the predicate.
        """
        exponent = self._exponent(adjoint)
        if (len(system), len(ancilla), len(resource)) != (
            self.system_bitsize,
            self.ancilla_bitsize,
            self.resource_bitsize,
        ):
            raise ValueError("Phase ports must match declared widths")
        wires, _ = bind_control_registers(controls, control_registers)
        payload = (*system, *ancilla, *resource)
        _validate_block_work(
            (*wires, *payload, *workspace),
            control_workspace,
            (*classical_bits, *control_classical_bits),
        )
        required = (
            max((*wires, *payload, *workspace, *control_workspace), default=-1) + 1
        )

        def emit(predicate: int) -> tuple[Instruction, ...]:
            scalar = global_phase_ir(exponent, controls=(predicate,), values=(1,))
            child = _capture_block(
                bind_child(predicate, adjoint),
                required,
                {predicate, *payload, *workspace},
            )
            # The original Controlled visitor visits the block before the
            # disconnected scalar, also when controlling Phase.adjoint().
            return (*child, *scalar)

        gates = controlled_component_ir(
            controls, control_registers, emit, control_workspace, control_classical_bits
        )
        return _BoundBlockOperation(gates, required)

    def make_on(
        self, child: _CircuitOperation, *, anchor: int, adjoint: bool = False
    ) -> "_BoundPhaseBlockEncoding":
        """Bind a directional child and an existing scalar carrier for append.

        Supply a child already bound in the requested direction. Symbolic
        phase metadata is valid for models but cannot be emitted numerically.
        """
        gates = global_phase_ir(self._exponent(adjoint), anchor=anchor)
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
