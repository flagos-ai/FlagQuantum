# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rz_via_phase_gradient.ipynb.

"""Bind the original gradient Rz body to caller-owned angle and gradient words."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.phase.rz_phase_gradient import (
    rz_via_phase_gradient_ir,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_controlled_add_or_subtract_resource_formula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class GradientRzExample:
    angle_width: SymbolicInt = 4
    gradient_width: SymbolicInt = 4
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("q", quantum_bit_type()),
                Register(
                    "angle",
                    fixed_point_register_type(self.angle_width, self.angle_width),
                ),
                Register(
                    "phase_grad",
                    fixed_point_register_type(self.gradient_width, self.gradient_width),
                ),
            )
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "RzViaPhaseGradient"
        child = (
            f"ControlledAddOrSubtract(QUInt({self.angle_width}), "
            f"QUInt({self.gradient_width}), add_on_one={not self.adjoint})"
        )
        return symbolic_call_graph(root, {root: {child: 1}}, max_depth=1)

    def resource_model(self) -> SymbolicResourceFormula:
        return source_controlled_add_or_subtract_resource_formula(
            self.angle_width,
            self.gradient_width,
            add_when_control_is_on=not self.adjoint,
        )

    @property
    def qubit_count(self) -> int:
        a, b = self.angle_width, self.gradient_width
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("Bind the Rz word widths before appending")
        return a + 2 * b

    def append_to(self, circuit: Circuit) -> Circuit:
        width = self.qubit_count
        a, b = self.angle_width, self.gradient_width
        assert isinstance(a, int) and isinstance(b, int)
        angle = tuple(range(1, 1 + a))
        gradient = tuple(range(1 + a, 1 + a + b))
        work = tuple(range(1 + a + b, width))
        gates = rz_via_phase_gradient_ir(
            0,
            angle,
            gradient,
            work,
            tuple(range(len(work))),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
