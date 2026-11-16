# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 unary_iteration.ipynb and its test fixtures.

"""Native bounded unary-iteration examples for the locked tutorial."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
    indexed_target_component_ir,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.compiler.registers.lookup.unary_iteration import (
    append_bounded_unary_tree,
)
from flagquantum.core.ir import CircuitIR, Instruction


@dataclass(frozen=True)
class UnaryXExample:
    """Apply X to the source's reverse-ordered target bank."""

    selection_width: int = 3
    targets: int = 5
    controls: int = 1

    @property
    def logical_width(self) -> int:
        return self.controls + self.selection_width + self.targets

    @property
    def qubit_count(self) -> int:
        return self.logical_width + indexed_component_workspace(
            (self.selection_width,), self.controls
        )

    def resource_model(self) -> ComponentCost:
        return indexed_component_cost(
            (self.selection_width,),
            (self.targets,),
            (ComponentCost("CNOT", 2, GateCost(clifford=1)),) * self.targets,
            target_width=self.targets,
            child_target_width=1,
            control_count=self.controls,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        start = self.controls + self.selection_width
        work = tuple(range(self.logical_width, self.qubit_count))
        gates = indexed_target_component_ir(
            (tuple(range(self.controls, start)),),
            (self.targets,),
            tuple(reversed(range(start, self.logical_width))),
            lambda _index, predicate, target: (Instruction("cx", (predicate, target)),),
            controls=tuple(range(self.controls)),
            workspace=work,
            classical_bits=tuple(range(len(work))),
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


class UnaryIntervalExample:
    """Toggle one of four targets over the original two nonzero intervals."""

    qubit_count = 15

    def resource_model(self) -> ComponentCost:
        # Outer selector uses a live input as predicate. Each inner two-row
        # subtree needs one And and its release, one switch CX and two leaf CX.
        return ComponentCost(
            "Unary intervals [3,5) and [6,8)",
            11,
            GateCost(and_bloq=2, clifford=10, measurement=2),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        gates: list[Instruction] = []

        def outer(i: int, predicate: int) -> None:
            def inner(j: int, active: int) -> None:
                gates.append(Instruction("cx", (active, 2 * (i - 3) + j)))

            append_bounded_unary_tree(
                gates, predicate, (3, 4, 5), 8, (12, 13, 14), (2, 3, 4), inner, lower=6
            )

        append_bounded_unary_tree(
            gates, None, (0, 1, 2), 5, (10, 11), (0, 1), outer, lower=3
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, tuple(gates)))
