# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/text_book_qpe.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Compose a named powered unitary, phase-window preparation and inverse QFT."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.layout.partition import partition_register
from ...compiler.registers.phase.phase_estimation_component import (
    rectangular_z_phase_estimation_program,
)
from ...compiler.registers.primitives.circuit_binding import bind_primitive_circuit
from ...core.ir import Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from .black_box_prepare import BlackBoxPrepare, PrepareEmitter, _concrete_layout
from .phase_estimation_window import PhaseEstimationWindow, rectangular_window_program
from .qft import qft
from .qft_resource_model import source_textbook_qft_cost, source_textbook_qft_signature

PoweredUnitaryEmitter = Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], int, bool],
    tuple[Instruction, ...],
]
_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class TextbookPhaseEstimation:
    """Keep exact source power boundaries and all original register descriptors.

    The callback receives control, named target ports, work, result slots,
    positive repetition count and direction. It owns the unitary's source
    power specialization. Generic powers may repeat a body, while fast-forward
    gates may combine exponents. Preparation is bound explicitly at execution.
    """

    unitary_signature: Signature
    ctrl_state_prep: PhaseEstimationWindow
    unitary: PoweredUnitaryEmitter
    preparation: BlackBoxPrepare | None = None
    qft_inverse: PrepareEmitter | None = None
    qft_inv_signature: Signature | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.m_bits, int)
            and self.inverse_qft_signature.n_qubits() != self.m_bits
        ):
            raise ValueError("Inverse QFT signature must match the phase width")

    @property
    def m_bits(self) -> SymbolicInt:
        return self.ctrl_state_prep.m_bits

    @property
    def target_registers(self) -> tuple[Register, ...]:
        return tuple(self.unitary_signature)

    @property
    def phase_registers(self) -> tuple[Register, ...]:
        return tuple(self.ctrl_state_prep.signature)

    @property
    def signature(self) -> Signature:
        return Signature((*self.phase_registers, *self.target_registers))

    @property
    def inverse_qft_signature(self) -> Signature:
        return (
            self.qft_inv_signature
            if self.qft_inv_signature is not None
            else source_textbook_qft_signature(self.m_bits)
        )

    def source_graph(
        self,
        *,
        root: _Node,
        preparation: _Node,
        controlled_unitary: _Node,
        inverse_qft: _Node,
    ) -> SymbolicCallGraph[_Node]:
        """Retain the source's non-fast-forward declared control count."""
        return symbolic_call_graph(
            root,
            {
                root: {
                    preparation: 1,
                    controlled_unitary: 2**self.m_bits - 1,
                    inverse_qft: 1,
                }
            },
        )

    def resource_model(
        self,
        preparation: ComponentCost,
        controlled_unitary: ComponentCost,
        inverse_qft: ComponentCost | None = None,
    ) -> ComponentCost:
        """Price declared children independently from physical power specialization."""
        if not isinstance(self.m_bits, int) or self.m_bits < 1:
            raise ValueError("Concrete QPE resources require a positive resolved width")
        target = sum(width for _, width in _concrete_layout(self.target_registers))
        transform = (
            inverse_qft
            if inverse_qft is not None
            else source_textbook_qft_cost(self.m_bits)
        )
        return ComponentCost(
            "TextbookPhaseEstimation",
            self.m_bits + target,
            calls=(
                CostCall(preparation, bystander_qubits=target),
                CostCall(
                    controlled_unitary,
                    2**self.m_bits - 1,
                    bystander_qubits=self.m_bits - 1,
                ),
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
        unitary_workspace: tuple[int, ...] = (),
        unitary_classical_bits: tuple[int, ...] = (),
        qft_workspace: tuple[int, ...] = (),
        qft_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Append the source's reversed-bit power schedule with explicit child work.

        Work may be reused between sequential children. Each child must restore
        its work independently. Outputs start clean for forward preparation.
        An explicit reverse emitter supplies the mathematical inverse, including
        measured state preparation. No final measurement is added by the source.
        """
        if (
            not isinstance(self.m_bits, int)
            or self.m_bits < 1
            or len(phase) != self.m_bits
        ):
            raise ValueError("QPE binding requires its positive concrete phase width")
        if type(adjoint) is not bool or self.preparation is None:
            raise ValueError(
                "QPE binding requires a Boolean direction and bound PREPARE"
            )
        if any(reg.side != RegisterSide.THRU for reg in self.target_registers):
            raise ValueError("Physical unitary target registers must be through ports")
        if (
            self.preparation.selection_width != len(phase)
            or self.preparation.junk_width
        ):
            raise ValueError("QPE preparation must bind only the declared phase output")
        ports = partition_register(target, _concrete_layout(self.target_registers))
        for work, bits in (
            (preparation_workspace, preparation_classical_bits),
            (unitary_workspace, unitary_classical_bits),
            (qft_workspace, qft_classical_bits),
        ):
            _validate_block_work((*phase, *target), work, bits)
        required = (
            max(
                (
                    *phase,
                    *target,
                    *preparation_workspace,
                    *unitary_workspace,
                    *qft_workspace,
                )
            )
            + 1
        )

        def capture(
            gates: tuple[Instruction, ...], allowed: tuple[int, ...]
        ) -> tuple[Instruction, ...]:
            return _capture_block(
                _BoundBlockOperation(gates, required), required, set(allowed)
            )

        prepare = capture(
            self.preparation.emit(
                phase,
                (),
                preparation_workspace,
                preparation_classical_bits,
                adjoint=adjoint,
            ),
            (*phase, *preparation_workspace),
        )
        powers: list[Instruction] = []
        schedule = tuple(enumerate(reversed(phase)))
        for index, control in reversed(schedule) if adjoint else schedule:
            powers.extend(
                capture(
                    self.unitary(
                        control,
                        ports.copy(),
                        unitary_workspace,
                        unitary_classical_bits,
                        2**index,
                        adjoint,
                    ),
                    (control, *target, *unitary_workspace),
                )
            )
        if self.qft_inverse is None:
            transform = bind_primitive_circuit(
                qft(len(phase), inverse=not adjoint).to_ir(), phase
            )
        else:
            qft_ports = partition_register(
                phase, _concrete_layout(tuple(self.inverse_qft_signature))
            )
            transform = capture(
                self.qft_inverse(qft_ports, qft_workspace, qft_classical_bits, adjoint),
                (*phase, *qft_workspace),
            )
        gates = (
            (*transform, *powers, *prepare)
            if adjoint
            else (*prepare, *powers, *transform)
        )
        return _BoundBlockOperation(gates, required)


def textbook_z_phase_estimation_program(width: int, exponent: float) -> ProgramRecord:
    """Store source Z-power QPE through the shared controlled-power graph owner."""
    if type(width) is not int or width < 1:
        raise ValueError("Stored phase estimation requires a positive width")
    return rectangular_z_phase_estimation_program(
        rectangular_window_program(width), exponent
    )
