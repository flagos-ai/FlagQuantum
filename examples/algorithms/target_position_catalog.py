# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 apply_gate_to_lth_target.py.

"""Native target-position bindings for the locked indexed gate tutorial."""

from dataclasses import dataclass
from typing import Literal

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
    indexed_pauli_targets_ir,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

Pauli = Literal["i", "x", "y", "z"]


@dataclass(frozen=True)
class TargetPositionExample:
    """Bind one bounded selection word and an ordered Pauli target bank."""

    selection_width: int
    control_count: int
    paulis: tuple[Pauli, ...]
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        registers = []
        if self.control_count:
            registers.append(
                Register("control", opaque_register_type(self.control_count))
            )
        registers.extend(
            (
                Register(
                    "selection",
                    bounded_unsigned_register_type(
                        self.selection_width, len(self.paulis)
                    ),
                ),
                Register("target", opaque_register_type(len(self.paulis))),
            )
        )
        return Signature(registers)

    @property
    def wire_symbols(self) -> tuple[str, ...]:
        return (
            *("@",) * self.control_count,
            *("In",) * self.selection_width,
            *(pauli.upper() for pauli in self.paulis),
        )

    @property
    def qubit_count(self) -> int:
        data = self.control_count + self.selection_width + len(self.paulis)
        return data + indexed_component_workspace(
            (self.selection_width,), self.control_count
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Reserve clean unary-iteration storage after the target bank."""
        target_start = self.control_count + self.selection_width
        data_end = target_start + len(self.paulis)
        width = self.qubit_count
        gates = indexed_pauli_targets_ir(
            (tuple(range(self.control_count, target_start)),),
            (len(self.paulis),),
            tuple(range(target_start, data_end)),
            self.paulis,
            controls=tuple(range(self.control_count)),
            workspace=tuple(range(data_end, width)),
            classical_bits=tuple(range(width - data_end)),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))

    def resource_model(self) -> ComponentCost:
        """Count controlled identity and Pauli leaves in the declared model."""
        children = tuple(
            ComponentCost(
                "I(2)" if label == "i" else f"C{label.upper()}",
                2,
                GateCost(clifford=int(label != "i")),
            )
            for label in self.paulis
        )
        return indexed_component_cost(
            (self.selection_width,),
            (len(self.paulis),),
            children,
            target_width=len(self.paulis),
            child_target_width=1,
            control_count=self.control_count,
            adjoint=self.adjoint,
        )


def apply_z_to_odd_call_graph() -> SymbolicCallGraph[str]:
    """Direct declaration of the locked two-control, four-target fixture."""
    root = "Apply Z to odd targets"
    return symbolic_call_graph(
        root,
        {
            root: {
                "And(1,1)": 1,
                "And(1,0)": 3,
                "I(2)": 2,
                "CNOT": 3,
                "CZ": 2,
                "And(1,1)†": 4,
            }
        },
        max_depth=1,
    )


def apply_z_to_odd_example() -> TargetPositionExample:
    return TargetPositionExample(3, 2, ("i", "z", "i", "z"))
