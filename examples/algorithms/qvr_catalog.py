# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 quantum_variable_rotation.ipynb.

"""Bind the source direct and gradient variable-rotation tutorial fixtures."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    qvr_phase_gradient_ir,
    source_qvr_phase_gradient_cost,
)
from flagquantum.compiler.registers.phase.qvr_zpow import (
    qvr_zpow_ir,
    source_qvr_zpow_cost,
    source_qvr_zpow_declaration,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class QvrExample:
    width: int = 12
    coefficient: float = 0.1
    epsilon: float = 1e-2
    fraction_bits: int | None = None
    via_gradient: bool = False
    adjoint: bool = False

    @property
    def fraction(self) -> int:
        return self.width if self.fraction_bits is None else self.fraction_bits

    @property
    def gradient_width(self) -> int:
        return (
            qvr_phase_format(
                self.width, self.fraction, self.coefficient, self.epsilon
            ).gradient_width
            if self.via_gradient
            else 0
        )

    @property
    def signature(self) -> Signature:
        ports = [Register("x", fixed_point_register_type(self.width, self.fraction))]
        if self.via_gradient:
            m = self.gradient_width
            ports.append(Register("phase_grad", fixed_point_register_type(m, m)))
        return Signature(tuple(ports))

    @property
    def qubit_count(self) -> int:
        if not self.via_gradient:
            return self.width
        n = self.width
        fmt = qvr_phase_format(n, self.fraction, self.coefficient, self.epsilon)
        m = fmt.gradient_width
        terms = scaled_phase_add_terms(
            n,
            self.fraction,
            m,
            self.coefficient,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
        work = max(
            (
                phase_add_workspace_size(len(range(n)[part]), m, right_shift=shift)
                for part, shift in terms
            ),
            default=0,
        )
        return n + m + work

    def call_graph(self) -> SymbolicCallGraph[str]:
        if self.via_gradient:
            name = "QvrPhaseGradient"
            fmt = qvr_phase_format(
                self.width, self.fraction, self.coefficient, self.epsilon
            )
            child = (
                f"AddScaledValIntoPhaseReg({self.width}, {self.fraction}, "
                f"{fmt.gradient_width}, {self.coefficient}, "
                f"{fmt.coefficient_width}, {fmt.coefficient_fraction_bits})"
            )
            return symbolic_call_graph(name, {name: {child: 1}}, max_depth=1)
        name = "QvrZPow"
        call, count = source_qvr_zpow_declaration(
            self.width, self.fraction, self.coefficient, self.epsilon
        )
        return symbolic_call_graph(
            name,
            {name: {f"ZPowGate({call.parameter}, {call.precision})": count}},
            max_depth=1,
        )

    def resource_model(self) -> ComponentCost:
        model = (
            source_qvr_phase_gradient_cost
            if self.via_gradient
            else source_qvr_zpow_cost
        )
        return model(self.width, self.fraction, self.coefficient, self.epsilon)

    def append_to(self, circuit: Circuit) -> Circuit:
        cost = tuple(range(self.width))
        boundary = self.width + self.gradient_width
        if self.via_gradient:
            work = tuple(range(boundary, self.qubit_count))
            gates = qvr_phase_gradient_ir(
                cost,
                tuple(range(self.width, boundary)),
                work,
                tuple(range(len(work))),
                cost_fraction_bits=self.fraction,
                coefficient=self.coefficient,
                epsilon=self.epsilon,
                adjoint=self.adjoint,
            )
        else:
            gates = qvr_zpow_ir(
                cost,
                cost_fraction_bits=self.fraction,
                coefficient=self.coefficient,
                epsilon=self.epsilon,
                adjoint=self.adjoint,
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
