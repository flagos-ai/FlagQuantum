# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 zpow_via_phase_gradient.ipynb.

"""Bind the original constant Z-power phase-gradient tutorial fixtures."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient import (
    source_zpow_gradient_load_constant,
    source_zpow_gradient_resource_formula,
    zpow_phase_gradient_width,
    zpow_via_phase_gradient_ir,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class GradientZPowExample:
    exponent: SymbolicReal
    width: SymbolicInt
    adjoint: bool = False

    @classmethod
    def from_precision(
        cls, exponent: SymbolicReal, *, epsilon: SymbolicReal
    ) -> "GradientZPowExample":
        return cls(exponent, zpow_phase_gradient_width(epsilon))

    @property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("q", quantum_bit_type()),
                Register(
                    "phase_grad", fixed_point_register_type(self.width, self.width)
                ),
            )
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "ZPowConstViaPhaseGradient"
        constant = source_zpow_gradient_load_constant(self.exponent, self.width)
        calls = {
            f"ControlledXorK({self.width}, {constant})": 2,
            f"AddIntoPhaseGrad({self.width}, {self.width}, sign={-1 if self.adjoint else 1})": 1,
        }
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def resource_model(self) -> SymbolicResourceFormula:
        return source_zpow_gradient_resource_formula(self.exponent, self.width)

    @property
    def qubit_count(self) -> int:
        if not isinstance(self.width, int):
            raise ValueError("Bind a concrete Z-power gradient width before appending")
        return 3 * self.width

    def append_to(self, circuit: Circuit) -> Circuit:
        n, exponent = self.width, self.exponent
        if not isinstance(n, int) or not isinstance(exponent, (int, float)):
            raise ValueError("Bind the Z-power width and angle before appending")
        gates = zpow_via_phase_gradient_ir(
            0,
            tuple(range(1, n + 1)),
            tuple(range(n + 1, 2 * n + 1)),
            tuple(range(2 * n + 1, 3 * n)),
            tuple(range(n - 1)),
            exponent=exponent,
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
