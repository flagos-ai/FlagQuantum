# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 prepare_uniform_superposition.ipynb.

"""Bind the locked uniform-preparation tutorial examples."""

from dataclasses import dataclass
from math import acos

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import (
    SymbolicInt,
    bit_length,
    floor,
    log2,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    _odd_factor,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_resource_formula,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class UniformPreparationExample:
    count: SymbolicInt = 5
    controls: tuple[int, ...] = ()
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return Signature(
            tuple(
                Register(
                    name,
                    quantum_bit_type() if width == 1 else opaque_register_type(width),
                )
                for name, width in (
                    ("ctrl", len(self.controls)),
                    ("target", bit_length(self.count - 1)),
                )
                if width != 0
            )
        )

    @property
    def qubit_count(self) -> int:
        if not isinstance(self.count, int):
            raise ValueError(
                "Bind a concrete number of states before appending uniform preparation"
            )
        width, work, _ = uniform_preparation_requirements(
            self.count, len(self.controls)
        )
        return len(self.controls) + width + work

    def resource_model(self) -> SymbolicResourceFormula:
        return source_uniform_preparation_resource_formula(
            self.count, control_values=self.controls
        )

    def diagram_symbols(self) -> tuple[str, ...]:
        if not isinstance(self.count, int):
            raise ValueError("Uniform diagram requires a concrete count")
        width = (self.count - 1).bit_length()
        symbols = tuple("@" if ctrl else "@(0)" for ctrl in self.controls)
        return (*symbols, f"UNIFORM({self.count})", *("target",) * (width - 1))

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "PrepareUniformSuperposition"
        count = self.count
        calls: dict[str, SymbolicInt]
        if isinstance(count, int):
            width = (count - 1).bit_length()
            odd = _odd_factor(count)
            prefix = (odd - 1).bit_length()
            calls = {"Hadamard": 2 * prefix}
            initial = (
                "CHadamard"
                if self.controls == (1,)
                else (
                    f"ControlledHadamard{self.controls}"
                    if self.controls
                    else "Hadamard"
                )
            )
            calls[initial] = calls.get(initial, 0) + width
            if prefix:
                angle = acos(1 - (1 << (prefix - 1)) / odd)
                predicate = (0,) * prefix + self.controls
                and_name = "And" if len(predicate) == 2 else "MultiAnd"
                calls.update(
                    {
                        "Allocate": 1,
                        "Free": 1,
                        f"LessThanConstant({prefix}, {odd})": 2,
                        f"Rz({angle}, 1e-11)": 2,
                        f"{and_name}{predicate}": 1,
                        f"{and_name}{predicate}†": 1,
                    }
                )
        else:
            from sympy import acos as symbolic_acos

            width = bit_length(count - 1)
            angle = symbolic_acos(1 - 2 ** floor(log2(count)) / count)
            compare = f"LessThanConstant({width}, {count})"
            calls = {
                "Hadamard": 3 * width,
                compare: 1,
                f"{compare}†": 1,
                f"Rz({angle}, 1e-11)": 2,
                f"MultiAnd(HasLength({width}))": 1,
                f"MultiAnd(HasLength({width}))†": 1,
            }
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(self.count, int):
            raise ValueError(
                "Bind a concrete number of states before appending uniform preparation"
            )
        c = len(self.controls)
        width, work, slots = uniform_preparation_requirements(self.count, c)
        body = uniform_preparation_ir(
            self.count,
            tuple(range(c, c + width)),
            tuple(range(c + width, c + width + work)),
            tuple(range(slots)),
            controls=tuple(range(c)),
            control_values=self.controls,
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, body))
