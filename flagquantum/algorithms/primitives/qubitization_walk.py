# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QubitizationWalkOperator.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qubitization/qubitization_walk_operator.py

"""Compose a SELECT block and reflection about its prepared signal."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.partition import RegisterLayout, unpartition_register
from ...compiler.registers.prepare_reflection import (
    prepare_reflection_ir,
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.ir import CircuitIR, Instruction
from .select_block_encoding import SelectBlockEncoding


@dataclass(frozen=True)
class QubitizationWalk:
    """Walk in the prepared signal basis of a SELECT-only block encoding."""

    block_encoding: SelectBlockEncoding

    @property
    def selection_layout(self) -> RegisterLayout:
        """Return the shared named selection ports in a deterministic order."""
        block = self.block_encoding
        select = dict(block.selection_layout)
        signal = dict(block.signal_state.selection_layout)
        if any(select[name] != signal[name] for name in select.keys() & signal.keys()):
            raise ValueError(
                "Walk selection ports with a shared name must have equal widths"
            )
        if (select.keys() | signal.keys()) & {name for name, _ in block.target_layout}:
            raise ValueError("Walk selection and target port names must be distinct")
        return tuple((select | signal).items())

    @property
    def alpha(self) -> float:
        """Return the L1 normalization of the underlying block."""
        return self.block_encoding.alpha

    def resource_model(
        self,
        select_cost: ComponentCost,
        prepare_cost: ComponentCost,
        unprepare_cost: ComponentCost,
        *,
        adjoint: bool = False,
    ) -> ComponentCost:
        """Count the ordered block and reflection calls with their bystanders."""
        block = self.block_encoding
        width = sum(size for _, size in self.selection_layout) + block.system_bitsize
        select = block.resource_model(select_cost)
        reflection = source_uncontrolled_prepare_reflection_cost(
            block.ancilla_bitsize, block.resource_bitsize, prepare_cost, unprepare_cost
        )
        calls = (
            CostCall(select, bystander_qubits=width - select.qubits),
            CostCall(reflection, bystander_qubits=width - block.ancilla_bitsize),
        )
        return ComponentCost(
            "QubitizationWalkOperator†" if adjoint else "QubitizationWalkOperator",
            width,
            calls=calls[::-1] if adjoint else calls,
        )

    def make_on(
        self,
        *,
        selection: Mapping[str, tuple[int, ...]],
        target: tuple[int, ...],
        signal_junk: tuple[int, ...] = (),
        select_workspace: tuple[int, ...] = (),
        select_classical_bits: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...] = (),
        prepare_classical_bits: tuple[int, ...] = (),
        reflection_workspace: tuple[int, ...],
        reflection_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> "_BoundQubitizationWalk":
        """Bind one walk step for circuit.append using caller-owned clean work.

        SELECT may reuse reflection work or signal junk between stages.
        Preparation and the zero mark can share work other than the phase
        target, which is reserved throughout reflection.
        """
        if type(adjoint) is not bool:
            raise ValueError("Walk direction must be boolean")
        block = self.block_encoding
        signal = block.signal_state
        logical = (*unpartition_register(selection, self.selection_layout), *target)
        select_wires = tuple(
            wire for name, _ in block.selection_layout for wire in selection[name]
        )
        signal_wires = tuple(
            wire for name, _ in signal.selection_layout for wire in selection[name]
        )
        if logical or select_workspace:
            _validate_wires((*logical, *select_workspace))
        _validate_wires((*logical, *signal_junk, *reflection_workspace))
        _validate_wires(
            (*logical, *signal_junk, *reflection_workspace[:1], *prepare_workspace)
        )
        select = block.emit(
            select_wires,
            target,
            select_workspace,
            select_classical_bits,
            adjoint=adjoint,
        )
        prepare = signal.emit(
            signal_wires, signal_junk, prepare_workspace, prepare_classical_bits
        )
        unprepare = signal.emit(
            signal_wires,
            signal_junk,
            prepare_workspace,
            prepare_classical_bits,
            adjoint=True,
        )
        reflection = prepare_reflection_ir(
            signal_wires,
            reflection_workspace,
            reflection_classical_bits,
            prepare=prepare,
            unprepare=unprepare,
            global_phase=-1,
        )
        gates = (*reflection, *select) if adjoint else (*select, *reflection)
        return _BoundQubitizationWalk(
            gates,
            max(
                (
                    *logical,
                    *signal_junk,
                    *select_workspace,
                    *prepare_workspace,
                    *reflection_workspace,
                )
            )
            + 1,
        )


@dataclass(frozen=True)
class _BoundQubitizationWalk:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Walk binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
