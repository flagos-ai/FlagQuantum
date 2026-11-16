# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 multi_control_multi_target_pauli.ipynb.

"""Source signatures, declared calls and append fixtures for Pauli controls and fanout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.bits.bitwise import fanout_ir
from flagquantum.compiler.registers.control.controlled_component_resource_model import (
    controlled_pattern_resource_formula,
)
from flagquantum.compiler.registers.control.multi_control import (
    multi_control_x_ir,
    multi_control_y_ir,
    multi_control_z_ir,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

Pauli = Literal["x", "y", "z"]


@dataclass(frozen=True)
class ControlledPauliExample:
    """Place controls before the target and borrow clean work after it."""

    pauli: Pauli
    activation: tuple[int, ...] | SymbolicInt

    @property
    def width(self) -> SymbolicInt:
        return (
            len(self.activation)
            if isinstance(self.activation, tuple)
            else self.activation
        )

    @property
    def signature(self) -> Signature:
        bit = quantum_bit_type()
        controls = (
            (Register("controls", bit, (self.width,)),) if self.width != 0 else ()
        )
        return Signature((*controls, Register("target", bit)))

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = f"MultiControl{self.pauli.upper()}"
        child = (
            self.pauli.upper()
            if self.width == 0
            else f"ControlledViaAnd({self.pauli.upper()})"
        )
        return symbolic_call_graph(root, {root: {child: 1}}, max_depth=1)

    def resource_model(self) -> SymbolicResourceFormula:
        if self.width == 0:
            return SymbolicResourceFormula(0, 1, 0, 1)
        return controlled_pattern_resource_formula(
            {
                "x": CliffordCall("cx", 2),
                "y": CliffordCall("cy", 2),
                "z": CliffordCall("cz", 2),
            }[self.pauli],
            SymbolicResourceFormula(0, 1, 0, 2),
            self.activation,
            target_width=1,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(self.activation, tuple):
            raise ValueError("Pauli append requires a concrete activation")
        n = len(self.activation)
        emit = {
            "x": multi_control_x_ir,
            "y": multi_control_y_ir,
            "z": multi_control_z_ir,
        }[self.pauli]
        work = tuple(range(n + 1, 2 * n))
        if circuit.n_qubits < n + 1 + len(work):
            raise ValueError("Pauli append requires its clean predicate workspace")
        body = emit(tuple(range(n)), self.activation, n, work, tuple(range(len(work))))
        return circuit.compose(
            CircuitIR(circuit.n_qubits, body, metadata={"batch_size": circuit.bsz})
        )


@dataclass(frozen=True)
class FanoutCNOTExample:
    """Use the fanout tree on arbitrary targets without allocating work."""

    targets: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature.build(control=1, targets=self.targets)

    def wire_symbols(self) -> tuple[str, ...]:
        """Give the source per-wire labels for a concrete fanout diagram."""
        if not isinstance(self.targets, int):
            raise ValueError(f"Symbolic bitsize {self.targets} not supported")
        return ("@", *("X",) * self.targets)

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "MultiTargetCNOT"
        calls: Mapping[str, SymbolicInt] = (
            {"CNOT": 2 * self.targets - 1} if isinstance(self.targets, int) else {}
        )
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        n = self.targets
        if not isinstance(n, int) or n < 1:
            raise ValueError("Fanout append requires a positive concrete target width")
        if circuit.n_qubits < n + 1:
            raise ValueError("Fanout append requires its control and target wires")
        body = fanout_ir(0, tuple(range(1, n + 1)))
        return circuit.compose(
            CircuitIR(circuit.n_qubits, body, metadata={"batch_size": circuit.bsz})
        )
