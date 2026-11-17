# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 StatePreparationViaRotations.

"""Bind the dense rotation examples and their phase-gradient ports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import pi
from typing import cast

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.rotation_resource_model import (
    rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_calls,
)
from flagquantum.algorithms.primitives.symbolic_rotation_resource_model import (
    source_rotation_preparation_signature as rotation_signature,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import RotationCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.phase.phase_gradient_lookup import (
    PhaseGradientLookupCall,
)
from flagquantum.core.ir import CircuitIR
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
        root = "StatePreparationViaRotations"
        calls: Counter[str] = Counter()
        direct = source_rotation_preparation_calls(
            len(self.coefficients),
            self.phase_width,
            self.controls,
            coefficients=self.coefficients,
            uncompute=self.uncompute,
        )
        for child, count in direct.items():
            if isinstance(child, PhaseGradientLookupCall):
                table = child.table
                assert table.datasets is not None
                name = (
                    f"PRGA({table.selection_bitsizes[0]} selection, "
                    f"{table.target_bitsizes[0]} phase, {table.num_controls} controls; "
                    f"{table.datasets[0]})"
                )
            elif isinstance(child, RotationCall):
                name = "Rx(+π/2)" if child.parameter == pi / 2 else "Rx(-π/2)"
            else:
                name = "X"
            calls[name] = cast(int, count)
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
