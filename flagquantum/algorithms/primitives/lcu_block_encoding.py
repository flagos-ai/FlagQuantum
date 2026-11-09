# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LCUBlockEncoding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/lcu_block_encoding.py

"""Conjugate a SELECT oracle by its preparation in the zero-signal basis."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, partition_register
from ...core.ir import CircuitIR, Instruction
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .select_block_encoding import ControlledSelectEmitter, SelectBlockEncoding


@dataclass(frozen=True)
class LCUBlockEncoding:
    """Bind a zero-signal block on selection, through junk and target wires.

    The wrapped SELECT block supplies the preparation used in the conjugation.
    For a controlled block, the specialized emitter receives the activation
    value and the single wire named `control`. Preparation stays unconditional.
    """

    select_block: SelectBlockEncoding
    control_value: int | None = None
    controlled_select: ControlledSelectEmitter | None = None

    def __post_init__(self) -> None:
        if self.control_value is not None and (
            type(self.control_value) is not int or self.control_value not in (0, 1)
        ):
            raise ValueError("LCU control value must be zero or one")
        if (self.control_value is None) != (self.controlled_select is None):
            raise ValueError(
                "A controlled LCU block requires a specialized SELECT emitter"
            )
        block = self.select_block
        prepare = block.signal_state
        prep_layout = (*prepare.selection_layout, *prepare.junk_layout)
        ports = dict(prep_layout)
        targets = dict(block.target_layout)
        if len(ports) != len(prep_layout) or ports.keys() & targets.keys():
            raise ValueError("LCU preparation and target port names must be distinct")
        if any(ports.get(name) != width for name, width in block.selection_layout):
            raise ValueError("LCU SELECT ports must match named preparation registers")
        if self.control_value is not None and "control" in (
            ports.keys() | targets.keys()
        ):
            raise ValueError("Controlled LCU reserves the control port name")

    @property
    def selection_layout(self) -> RegisterLayout:
        return self.select_block.signal_state.selection_layout

    @property
    def junk_layout(self) -> RegisterLayout:
        return self.select_block.signal_state.junk_layout

    @property
    def target_layout(self) -> RegisterLayout:
        return self.select_block.target_layout

    @property
    def ancilla_bitsize(self) -> int:
        return self.select_block.ancilla_bitsize

    @property
    def resource_bitsize(self) -> int:
        return self.select_block.resource_bitsize

    @property
    def system_bitsize(self) -> int:
        return self.select_block.system_bitsize

    @property
    def signal_state(self) -> BlackBoxPrepare:
        """The zero signal acts as identity on selection alone."""
        return BlackBoxPrepare(self.selection_layout, (), _identity_prepare)

    @property
    def alpha(self) -> float:
        return self.select_block.alpha

    @property
    def epsilon(self) -> float:
        return self.select_block.epsilon

    def resource_model(
        self,
        select_cost: ComponentCost,
        prepare_cost: ComponentCost,
        unprepare_cost: ComponentCost,
        *,
        adjoint: bool = False,
        controlled: bool = False,
    ) -> ComponentCost:
        """Count the conjugation, given the directional cost of SELECT."""
        block = self.select_block
        system = self.system_bitsize
        controls = int(self.control_value is not None or controlled)
        ports = self.ancilla_bitsize + self.resource_bitsize + system + controls
        select_width = sum(width for _, width in block.selection_layout)
        return ComponentCost(
            "LCUBlockEncoding†" if adjoint else "LCUBlockEncoding",
            ports,
            calls=(
                CostCall(prepare_cost, bystander_qubits=system + controls),
                CostCall(
                    select_cost,
                    bystander_qubits=ports - select_width - system - controls,
                ),
                CostCall(unprepare_cost, bystander_qubits=system + controls),
            ),
        )

    def make_on(
        self,
        *,
        selection: tuple[int, ...],
        junk: tuple[int, ...],
        target: tuple[int, ...],
        control: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...] = (),
        prepare_classical_bits: tuple[int, ...] = (),
        select_workspace: tuple[int, ...] = (),
        select_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> "_BoundLCUBlockEncoding":
        """Bind one block for circuit.append with explicit work and slots."""
        if type(adjoint) is not bool:
            raise ValueError("LCU direction must be boolean")
        if len(control) != int(self.control_value is not None):
            raise ValueError(
                "LCU control binding does not match its configured polarity"
            )
        logical = (*control, *selection, *junk, *target)
        for workspace in (prepare_workspace, select_workspace):
            if logical or workspace:
                _validate_wires((*logical, *workspace))
        ports = partition_register(selection, self.selection_layout)
        ports.update(partition_register(junk, self.junk_layout))
        block = self.select_block
        prepare = block.signal_state
        select_words = tuple(
            wire for name, _ in block.selection_layout for wire in ports[name]
        )
        if self.control_value is not None and self.controlled_select is not None:
            block = block.with_control(self.control_value, self.controlled_select)
        before = prepare.emit(
            selection, junk, prepare_workspace, prepare_classical_bits
        )
        middle = block.emit(
            (*control, *select_words),
            target,
            select_workspace,
            select_classical_bits,
            adjoint=adjoint,
        )
        after = prepare.emit(
            selection, junk, prepare_workspace, prepare_classical_bits, adjoint=True
        )
        required = (
            max((*logical, *prepare_workspace, *select_workspace), default=-1) + 1
        )
        return _BoundLCUBlockEncoding((*before, *middle, *after), required)


@dataclass(frozen=True)
class _BoundLCUBlockEncoding:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("LCU binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
