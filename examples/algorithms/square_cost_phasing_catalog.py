# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phasing_via_cost_function.ipynb.

"""Bind the tutorial's Square, phase oracle, and Square adjoint."""

from dataclasses import dataclass

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.arithmetic.integer_square import (
    integer_square_workspace_size,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_format,
    source_qvr_phase_gradient_cost,
)
from flagquantum.compiler.registers.phase.qvr_zpow import source_qvr_zpow_cost
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.compiler.registers.phase.square_cost_phasing import (
    square_cost_gradient_ir,
    square_cost_zpow_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class SquareCostPhasingExample:
    via_gradient: bool = False
    width: int = 5
    coefficient: float = 0.1234
    epsilon: float = 1e-8
    normalize: bool = True

    @property
    def fraction(self) -> int:
        return 2 * self.width if self.normalize else 0

    @property
    def gradient_width(self) -> int:
        return (
            qvr_phase_format(
                2 * self.width, self.fraction, self.coefficient, self.epsilon
            ).gradient_width
            if self.via_gradient
            else 0
        )

    @property
    def signature(self) -> Signature:
        ports = [Register("a", unsigned_register_type(self.width))]
        if self.via_gradient:
            m = self.gradient_width
            ports.append(Register("phase_grad", fixed_point_register_type(m, m)))
        return Signature(tuple(ports))

    @property
    def square_end(self) -> int:
        return 3 * self.width + integer_square_workspace_size(self.width)

    @property
    def qubit_count(self) -> int:
        n, m = 2 * self.width, self.gradient_width
        if not m:
            return self.square_end
        fmt = qvr_phase_format(n, self.fraction, self.coefficient, self.epsilon)
        terms = scaled_phase_add_terms(
            n,
            self.fraction,
            m,
            self.coefficient,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
        carries = max(
            (
                phase_add_workspace_size(len(range(n)[part]), m, right_shift=shift)
                for part, shift in terms
            ),
            default=0,
        )
        return self.square_end + m + carries

    def call_graph(self, max_depth: int | None = 1) -> SymbolicCallGraph[str]:
        """Show ordered stage identities and an optional resource-category expansion."""
        model = self.resource_model()
        square, phase, erase = (call.component for call in model.calls)
        counts = square.gates.toffoli
        phase_gates = phase.gates
        return symbolic_call_graph(
            model.name,
            {
                model.name: {square.name: 1, phase.name: 1, erase.name: 1},
                square.name: {"Toffoli": counts},
                erase.name: {"Toffoli": counts},
                phase.name: {
                    key: count
                    for key, count in (
                        ("Toffoli", phase_gates.toffoli),
                        ("Z power rotation", phase_gates.rotation),
                        ("T", phase_gates.t),
                        ("Clifford", phase_gates.clifford),
                    )
                    if count
                },
            },
            max_depth=max_depth,
        )

    def resource_model(self) -> ComponentCost:
        """Declare the same source stages and their logical live wires."""
        n, m = self.width, self.gradient_width
        square_gates = GateCost(toffoli=n * (n - 1))
        if self.via_gradient:
            phase = source_qvr_phase_gradient_cost(
                2 * n, self.fraction, self.coefficient, self.epsilon
            )
        else:
            phase = source_qvr_zpow_cost(
                2 * n, self.fraction, self.coefficient, self.epsilon
            )
        return ComponentCost(
            "PhasingViaCostFunction",
            n + m,
            calls=(
                CostCall(
                    ComponentCost("Square", 3 * n, square_gates), bystander_qubits=m
                ),
                CostCall(phase, bystander_qubits=n),
                CostCall(
                    ComponentCost("Square†", 3 * n, square_gates), bystander_qubits=m
                ),
            ),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        n = self.width
        data, result, square_work = (
            tuple(range(n)),
            tuple(range(n, 3 * n)),
            tuple(range(3 * n, self.square_end)),
        )
        if self.via_gradient:
            boundary = self.square_end + self.gradient_width
            work = tuple(range(boundary, self.qubit_count))
            gates = square_cost_gradient_ir(
                data,
                result,
                square_work,
                tuple(range(self.square_end, boundary)),
                work,
                tuple(range(len(work))),
                cost_fraction_bits=self.fraction,
                coefficient=self.coefficient,
                epsilon=self.epsilon,
            )
        else:
            gates = square_cost_zpow_ir(
                data,
                result,
                square_work,
                cost_fraction_bits=self.fraction,
                coefficient=self.coefficient,
                epsilon=self.epsilon,
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
