# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BlackBoxPrepare and its AutoPartition binding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/black_box_prepare.py

"""Expose a concrete preparation through packed selection and junk ports."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.partition import RegisterLayout, partition_register
from ...core.ir import Instruction

PrepareEmitter = Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    tuple[Instruction, ...],
]


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
        selection_ports = partition_register(selection, self.selection_layout)
        junk_ports = partition_register(junk, self.junk_layout)
        if selection_ports.keys() & junk_ports.keys():
            raise ValueError("Preparation selection and junk names must be distinct")
        allowed = (*selection, *junk, *workspace)
        if allowed:
            _validate_wires(allowed)
        if not selection:
            return ()
        gates = self.prepare(
            selection_ports | junk_ports, workspace, classical_bits, adjoint
        )
        qubits = set(allowed)
        if any(q not in qubits for gate in gates for q in gate.wires):
            raise ValueError("Preparation used a qubit outside the bound registers")
        return gates
