# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_gradient.ipynb.

"""Register bindings for the locked phase-gradient tutorial."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_phase_add_cost,
)
from flagquantum.compiler.registers.phase.phase_addition import (
    phase_add_ir,
    phase_add_workspace_size,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    phase_gradient_state_ir,
    phase_gradient_unitary_ir,
    source_phase_gradient_state_cost,
    source_phase_gradient_unitary_cost,
    source_symbolic_phase_gradient_cost,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
    source_scaled_phase_add_cost,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class GradientUnitaryExample:
    width: SymbolicInt = 4
    epsilon: float = 1e-10

    @property
    def signature(self) -> Signature:
        return Signature(
            (Register("phase_grad", fixed_point_register_type(self.width, self.width)),)
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "PhaseGradientUnitary"
        width = self.width
        if isinstance(width, int):
            calls = {
                f"ZPowGate({1 / 2**i}, {self.epsilon / width})": 1 for i in range(width)
            }
            return symbolic_call_graph(root, {root: calls}, max_depth=1)
        calls_symbolic = {f"ZPowGate({2 ** (-width)}, {self.epsilon / width})": width}
        return symbolic_call_graph(root, {root: calls_symbolic}, max_depth=1)

    def resource_model(self) -> ComponentCost | SymbolicResourceFormula:
        if isinstance(self.width, int):
            return source_phase_gradient_unitary_cost(self.width)
        return source_symbolic_phase_gradient_cost(self.width)

    @property
    def qubit_count(self) -> int:
        if not isinstance(self.width, int):
            raise ValueError("Bind a concrete phase-gradient width before appending")
        return self.width

    def append_to(self, circuit: Circuit) -> Circuit:
        gates = phase_gradient_unitary_ir(tuple(range(self.qubit_count)))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


@dataclass(frozen=True)
class GradientStateExample:
    width: int = 4

    @property
    def signature(self) -> Signature:
        return Signature(
            (
                Register(
                    "phase_grad",
                    fixed_point_register_type(self.width, self.width),
                    side=RegisterSide.RIGHT,
                ),
            )
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "PhaseGradientState"
        calls = {
            f"Allocate({self.width})": 1,
            f"OnEach({self.width}, Hadamard)": 1,
            f"PhaseGradientUnitary({self.width}, -1)": 1,
        }
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def resource_model(self) -> ComponentCost:
        return source_phase_gradient_state_cost(self.width)

    @property
    def qubit_count(self) -> int:
        return self.width

    def append_to(self, circuit: Circuit) -> Circuit:
        gates = phase_gradient_state_ir(tuple(range(self.width)), exponent=-1)
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


@dataclass(frozen=True)
class GradientAdditionExample:
    scaled: bool = False

    @property
    def width(self) -> int:
        return 2 if self.scaled else 4

    @property
    def signature(self) -> Signature:
        dtype = fixed_point_register_type(self.width, self.width)
        return Signature((Register("x", dtype), Register("phase_grad", dtype)))

    def call_graph(self) -> SymbolicCallGraph[str]:
        model = self.resource_model()
        root = model.name
        calls = (
            {"AddIntoPhaseGrad(2, 2)": model.calls[0].repetitions}
            if self.scaled
            else {"Toffoli": model.gates.toffoli}
        )
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def resource_model(self) -> ComponentCost:
        if self.scaled:
            return source_scaled_phase_add_cost(2, 2, 2, 2, 2, 2)
        return source_phase_add_cost(4, 4)

    @property
    def work_size(self) -> int:
        if not self.scaled:
            return phase_add_workspace_size(4, 4)
        return max(
            (
                phase_add_workspace_size(len(range(2)[word]), 2, right_shift=shift)
                for word, shift in scaled_phase_add_terms(2, 2, 2, 2, 2, 2)
            ),
            default=0,
        )

    @property
    def qubit_count(self) -> int:
        return 2 * self.width + self.work_size

    def append_to(self, circuit: Circuit) -> Circuit:
        width = self.width
        source = tuple(range(width))
        gradient = tuple(range(width, 2 * width))
        work = tuple(range(2 * width, self.qubit_count))
        feedback = tuple(range(len(work)))
        gates = (
            scaled_phase_add_ir(
                source,
                gradient,
                work,
                feedback,
                input_fraction_bits=2,
                coefficient=2,
                coefficient_width=2,
                coefficient_fraction_bits=2,
            )
            if self.scaled
            else phase_add_ir(source, gradient, work, feedback)
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
