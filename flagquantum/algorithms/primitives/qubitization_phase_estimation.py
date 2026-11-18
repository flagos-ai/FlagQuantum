# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/qubitization_qpe.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Compose the frozen reflection-sandwiched walk-power phase-estimation schedule."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...core.ir import Instruction
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ._block_binding import _BoundBlockOperation, _capture_block
from .black_box_prepare import BlackBoxPrepare, PrepareEmitter, _concrete_layout
from .phase_estimation_window import PhaseEstimationWindow
from .qft_resource_model import source_textbook_qft_cost
from .select_block_encoding import ControlledSelectEmitter
from .textbook_phase_estimation import TextbookPhaseEstimation

_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class QubitizationPhaseEstimation:
    """Preserve the source schedule, including its unresolved paper discrepancy.

    Child callbacks own independent directions and restore caller-supplied
    scratch. The reflection callback activates on zero. Only the lowest phase
    bit controls a walk. Other bits control two reflections around an
    unconditional repeated walk. All logical walk ports remain through ports.
    """

    walk_signature: Signature
    reflection_signature: Signature
    ctrl_state_prep: PhaseEstimationWindow
    walk: PrepareEmitter
    controlled_walk: ControlledSelectEmitter
    zero_controlled_reflection: ControlledSelectEmitter
    preparation: BlackBoxPrepare | None = None
    qft_inverse: PrepareEmitter | None = None
    qft_inv_signature: Signature | None = None

    def __post_init__(self) -> None:
        walk_ports = {reg.name: reg for reg in self.walk_signature}
        if any(walk_ports.get(reg.name) != reg for reg in self.reflection_signature):
            raise ValueError("Reflection registers must be unchanged walk registers")
        # Reuse the shared inverse-QFT signature check without emitting children.
        self._composition()

    @property
    def m_bits(self) -> SymbolicInt:
        return self.ctrl_state_prep.m_bits

    @property
    def target_registers(self) -> tuple[Register, ...]:
        return tuple(self.walk_signature)

    @property
    def phase_registers(self) -> tuple[Register, ...]:
        return tuple(self.ctrl_state_prep.signature)

    @property
    def signature(self) -> Signature:
        return self._composition().signature

    @property
    def inverse_qft_signature(self) -> Signature:
        return self._composition().inverse_qft_signature

    def __str__(self) -> str:
        return f"QubitizationPhaseEstimation[{self.m_bits}]"

    def _composition(self) -> TextbookPhaseEstimation:
        return TextbookPhaseEstimation(
            self.walk_signature,
            self.ctrl_state_prep,
            self._emit_power,
            self.preparation,
            self.qft_inverse,
            self.qft_inv_signature,
        )

    def _emit_power(
        self,
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        repetitions: int,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        if repetitions == 1:
            return self.controlled_walk(control, ports, work, slots, adjoint)
        reflected = {reg.name: ports[reg.name] for reg in self.reflection_signature}
        through = tuple(wire for word in ports.values() for wire in word)
        reflection = self.zero_controlled_reflection(
            control, reflected, work, slots, adjoint
        )
        required = 1 + max(control, *through, *work)
        reflection = _capture_block(
            _BoundBlockOperation(reflection, required),
            required,
            {control, *work, *(wire for word in reflected.values() for wire in word)},
        )
        body = self.walk(ports, work, slots, adjoint) * repetitions
        return (*reflection, *body, *reflection)

    def source_graph(
        self,
        *,
        root: _Node,
        preparation: _Node,
        controlled_walk: _Node,
        zero_controlled_reflection: _Node,
        walk: _Node,
        inverse_qft: _Node,
    ) -> SymbolicCallGraph[_Node]:
        """Retain all five original declared children and symbolic weights."""
        return symbolic_call_graph(
            root,
            {
                root: {
                    preparation: 1,
                    controlled_walk: 1,
                    zero_controlled_reflection: 2 * (self.m_bits - 1),
                    walk: 2**self.m_bits - 2,
                    inverse_qft: 1,
                }
            },
        )

    def resource_model(
        self,
        preparation: ComponentCost,
        controlled_walk: ComponentCost,
        zero_controlled_reflection: ComponentCost,
        walk: ComponentCost,
        inverse_qft: ComponentCost | None = None,
    ) -> ComponentCost:
        """Compose original child models independently from the emitted inventory."""
        if not isinstance(self.m_bits, int) or self.m_bits < 1:
            raise ValueError("Concrete qubitization QPE resources need positive width")
        target = sum(width for _, width in _concrete_layout(self.target_registers))
        reflected = sum(
            width for _, width in _concrete_layout(tuple(self.reflection_signature))
        )
        transform = (
            inverse_qft
            if inverse_qft is not None
            else source_textbook_qft_cost(self.m_bits)
        )
        return ComponentCost(
            "QubitizationPhaseEstimation",
            self.m_bits + target,
            calls=(
                CostCall(preparation, bystander_qubits=target),
                CostCall(controlled_walk, bystander_qubits=self.m_bits - 1),
                CostCall(
                    zero_controlled_reflection,
                    2 * (self.m_bits - 1),
                    bystander_qubits=self.m_bits + target - reflected - 1,
                ),
                CostCall(walk, 2**self.m_bits - 2, bystander_qubits=self.m_bits),
                CostCall(transform, bystander_qubits=target),
            ),
        )

    def make_on(
        self,
        *,
        phase: tuple[int, ...],
        target: tuple[int, ...],
        preparation_workspace: tuple[int, ...] = (),
        preparation_classical_bits: tuple[int, ...] = (),
        walk_workspace: tuple[int, ...] = (),
        walk_classical_bits: tuple[int, ...] = (),
        qft_workspace: tuple[int, ...] = (),
        qft_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Append the original schedule using the shared QPE binding owner."""
        return self._composition().make_on(
            phase=phase,
            target=target,
            preparation_workspace=preparation_workspace,
            preparation_classical_bits=preparation_classical_bits,
            unitary_workspace=walk_workspace,
            unitary_classical_bits=walk_classical_bits,
            qft_workspace=qft_workspace,
            qft_classical_bits=qft_classical_bits,
            adjoint=adjoint,
        )
