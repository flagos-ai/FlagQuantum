# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 apply_lth_bloq.py.

"""Native append for the locked shared-target T/H/Z/X tutorial fixture."""

from dataclasses import dataclass
from math import prod

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_ir,
    indexed_component_workspace,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.compiler.registers.phase.phase_family import phase_family_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class IndexedComponentExample:
    adjoint: bool = False
    shape: tuple[int, ...] = (4,)

    def __post_init__(self) -> None:
        if self.shape not in ((2,), (3,), (4,), (2, 2)):
            raise ValueError("Choose one of the original shared-target example shapes")

    @property
    def axis_widths(self) -> tuple[int, ...]:
        return tuple((extent - 1).bit_length() for extent in self.shape)

    @property
    def signature(self) -> Signature:
        selection = tuple(
            Register(
                "selection" if len(self.shape) == 1 else f"selection{i}",
                bounded_unsigned_register_type(width, extent),
            )
            for i, (width, extent) in enumerate(
                zip(self.axis_widths, self.shape, strict=True)
            )
        )
        return Signature(
            (
                Register("control", quantum_bit_type()),
                *selection,
                Register("q", quantum_bit_type()),
            )
        )

    @property
    def qubit_count(self) -> int:
        return (
            2 + sum(self.axis_widths) + indexed_component_workspace(self.axis_widths, 1)
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Dispatch the selected controlled leaf onto the shared target."""

        start = 1
        axes = []
        for width in self.axis_widths:
            axes.append(tuple(range(start, start + width)))
            start += width
        target = start

        def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
            if index == 0:
                return phase_family_ir(
                    "t",
                    target,
                    controls=(predicate,),
                    values=(1,),
                    adjoint=self.adjoint,
                )
            return (Instruction(("ch", "cz", "cx")[index - 1], (predicate, target)),)

        work = tuple(range(target + 1, self.qubit_count))
        gates = indexed_component_ir(
            tuple(axes),
            self.shape,
            emit,
            controls=(0,),
            workspace=work,
            classical_bits=tuple(range(len(work))),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))

    def resource_model(self) -> ComponentCost:
        """Use the original controlled Bloq leaf costs for these four gates."""
        children = (
            ComponentCost("C[T]", 2, GateCost(rotation=1)),
            CONTROLLED_HADAMARD_COST,
            ComponentCost("CZ", 2, GateCost(clifford=1)),
            ComponentCost("CNOT", 2, GateCost(clifford=1)),
        )
        return indexed_component_cost(
            self.axis_widths,
            self.shape,
            children[: prod(self.shape)],
            target_width=1,
            child_target_width=1,
            control_count=1,
            adjoint=self.adjoint,
        )


def indexed_component_example_call_graph() -> SymbolicCallGraph[str]:
    root = "Apply indexed component"
    return symbolic_call_graph(
        root,
        {
            root: {
                "C[T]": 1,
                "CHadamard": 1,
                "CZ": 1,
                "CNOT": 4,
                "And(1,0)": 3,
                "And(1,1)†": 3,
            }
        },
        max_depth=1,
    )
