# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 StatePreparationViaRotations.

"""Bind the dense rotation examples and their phase-gradient ports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.rotation_resource_model import (
    rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.rotation_tables import rotation_tables
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt, bit_length
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

ORIGINAL_COEFFICIENTS = (
    (-0.42677669529663675 - 0.1767766952966366j),
    (0.17677669529663664 - 0.4267766952966367j),
    (0.17677669529663675 - 0.1767766952966368j),
    (0.07322330470336305 - 0.07322330470336309j),
    (0.4267766952966366 - 0.17677669529663692j),
    (0.42677669529663664 + 0.17677669529663675j),
    (0.0732233047033631 + 0.17677669529663678j),
    (-0.07322330470336308 - 0.17677669529663678j),
)


def rotation_signature(
    count: SymbolicInt, phase_width: SymbolicInt, controls: int = 0
) -> Signature:
    return Signature(
        tuple(
            Register(name, opaque_register_type(width))
            for name, width in (
                ("prepare_control", controls),
                ("target_state", bit_length(count - 1)),
                ("phase_gradient", phase_width),
            )
            if width != 0
        )
    )


@dataclass(frozen=True)
class RotationPreparationExample:
    coefficients: tuple[complex, ...] = ORIGINAL_COEFFICIENTS
    phase_width: int = 2
    controls: int = 0
    uncompute: bool = False

    @property
    def preparation(self) -> RotationPreparation:
        return RotationPreparation(self.coefficients, self.phase_width)

    @property
    def target_width(self) -> int:
        return (len(self.coefficients) - 1).bit_length()

    @property
    def gradient(self) -> tuple[int, ...]:
        offset = self.controls + self.target_width
        return tuple(range(offset, offset + self.phase_width))

    @property
    def signature(self) -> Signature:
        return rotation_signature(
            len(self.coefficients), self.phase_width, self.controls
        )

    @property
    def qubit_count(self) -> int:
        work, _ = self.preparation.requirements(self.controls, uncompute=self.uncompute)
        return self.gradient[-1] + 1 + work

    def resource_model(self) -> ComponentCost:
        return rotation_preparation_cost(
            self.preparation, self.controls, uncompute=self.uncompute
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        rows, phases = rotation_tables(
            self.coefficients, self.phase_width, uncompute=self.uncompute
        )
        root = "StatePreparationViaRotations"
        calls = Counter(
            {
                "Rx(+π/2)": len(rows),
                "Rx(-π/2)": len(rows),
                "X": 2,
            }
        )
        for selection, words in enumerate((*rows, phases)):
            calls[
                f"PRGA({selection} selection, {self.phase_width} phase, {self.controls + 1} controls; {words})"
            ] += 1
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        work, slots = self.preparation.requirements(
            self.controls, uncompute=self.uncompute
        )
        start = self.gradient[-1] + 1
        gates = self.preparation.emit(
            tuple(range(self.controls)),
            tuple(range(self.controls, self.controls + self.target_width)),
            self.gradient,
            tuple(range(start, start + work)),
            tuple(range(slots)),
            uncompute=self.uncompute,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
