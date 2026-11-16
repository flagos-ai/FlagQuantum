# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 hamming_weight_phasing.ipynb.

"""Bind both locked four-bit Hamming phase examples."""

from dataclasses import dataclass
from math import pi

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_requirements,
)
from flagquantum.compiler.registers.phase.hamming_weight_phasing import (
    hamming_weight_gradient_phasing_ir,
    hamming_weight_phasing_ir,
    source_hamming_weight_gradient_phasing_cost,
    source_hamming_weight_phasing_cost,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.qvr_phase_gradient import qvr_phase_format
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class HammingPhasingExample:
    width: int = 4
    exponent: float = pi / 2
    epsilon: float = 1e-10
    via_gradient: bool = False
    adjoint: bool = False

    @property
    def gradient_width(self) -> int:
        if not self.via_gradient:
            return 0
        return qvr_phase_format(
            self.width.bit_length(), 0, self.exponent / 2, self.epsilon
        ).gradient_width

    @property
    def signature(self) -> Signature:
        data = Register("x", unsigned_register_type(self.width))
        if not self.via_gradient:
            return Signature((data,))
        m = self.gradient_width
        return Signature(
            (data, Register("phase_grad", fixed_point_register_type(m, m)))
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        """Declare the top-level source calls for the forward example."""
        n = self.width
        bits = n.bit_length()
        count = f"HammingWeightCompute({n})"
        calls = {count: 1, f"{count}†": 1}
        if self.via_gradient:
            name = "Hamming-weight phase gradient"
            calls[f"QvrPhaseGradient({bits}, {self.exponent / 2}, {self.epsilon})"] = 1
        else:
            name = "Hamming-weight phasing"
            calls[f"ZPow({self.exponent}, {self.epsilon / bits})"] = bits
        return symbolic_call_graph(name, {name: calls}, max_depth=1)

    def resource_model(self) -> ComponentCost:
        if self.via_gradient:
            return source_hamming_weight_gradient_phasing_cost(
                self.width, self.exponent, self.epsilon
            )
        return source_hamming_weight_phasing_cost(self.width, self.exponent)

    def addition_work_size(self) -> int:
        if not self.via_gradient:
            return 0
        width = self.width.bit_length()
        coefficient = self.exponent / 2
        fmt = qvr_phase_format(width, 0, coefficient, self.epsilon)
        terms = scaled_phase_add_terms(
            width,
            0,
            fmt.gradient_width,
            coefficient,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
        return max(
            (
                phase_add_workspace_size(
                    len(range(width)[word]), fmt.gradient_width, right_shift=shift
                )
                for word, shift in terms
            ),
            default=0,
        )

    @property
    def qubit_count(self) -> int:
        junk, count = hamming_weight_requirements(self.width)
        return (
            self.width + self.gradient_width + junk + count + self.addition_work_size()
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the phase body, consuming any caller-owned gradient state."""
        n = self.width
        junk_width, weight_width = hamming_weight_requirements(n)
        m = self.gradient_width
        j = n + m
        w = j + junk_width
        end = w + weight_width
        data = tuple(range(n))
        junk = tuple(range(j, w))
        weight = tuple(range(w, end))
        count_bits = tuple(range(junk_width))
        if self.via_gradient:
            addition_width = self.addition_work_size()
            gates = hamming_weight_gradient_phasing_ir(
                data,
                junk,
                weight,
                tuple(range(n, j)),
                tuple(range(end, end + addition_width)),
                count_bits,
                tuple(range(addition_width)),
                exponent=self.exponent,
                epsilon=self.epsilon,
                adjoint=self.adjoint,
            )
        else:
            gates = hamming_weight_phasing_ir(
                data,
                junk,
                weight,
                count_bits,
                exponent=self.exponent,
                adjoint=self.adjoint,
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))
