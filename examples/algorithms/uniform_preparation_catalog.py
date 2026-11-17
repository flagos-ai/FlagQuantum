# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 prepare_uniform_superposition.ipynb.

"""Bind the locked uniform-preparation tutorial examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    _odd_factor,
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_resource_model import (
    normalize_uniform_controls,
    source_uniform_abstract_calls,
    source_uniform_preparation_resource_formula,
    source_uniform_preparation_signature,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class UniformPreparationExample:
    count: SymbolicInt = 5
    controls: tuple[int, ...] = ()
    adjoint: bool = False
    abstract_control_width: SymbolicInt | None = None

    def __post_init__(self) -> None:
        values, width = normalize_uniform_controls(
            self.controls, self.abstract_control_width
        )
        object.__setattr__(self, "controls", values)
        object.__setattr__(self, "abstract_control_width", width)

    @property
    def signature(self) -> Signature:
        width = (
            len(self.controls)
            if self.abstract_control_width is None
            else self.abstract_control_width
        )
        return source_uniform_preparation_signature(self.count, width)

    @property
    def qubit_count(self) -> int:
        if not isinstance(self.count, int) or self.abstract_control_width is not None:
            raise ValueError(
                "Bind a concrete number of states before appending uniform preparation"
            )
        width, work, _ = uniform_preparation_requirements(
            self.count, len(self.controls)
        )
        return len(self.controls) + width + work

    def resource_model(self) -> SymbolicResourceFormula:
        return source_uniform_preparation_resource_formula(
            self.count,
            control_values=self.controls,
            abstract_control_width=self.abstract_control_width,
        )

    def diagram_symbols(self) -> tuple[str, ...]:
        if not isinstance(self.count, int) or self.abstract_control_width is not None:
            raise ValueError("Uniform diagram requires a concrete count")
        width = (self.count - 1).bit_length()
        symbols = tuple("@" if ctrl else "@(0)" for ctrl in self.controls)
        return (*symbols, f"UNIFORM({self.count})", *("target",) * (width - 1))

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "PrepareUniformSuperposition"
        count = self.count
        calls: dict[str, SymbolicInt]
        if isinstance(count, int) and self.abstract_control_width is None:
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
            calls = source_uniform_abstract_calls(count)
        return symbolic_call_graph(root, {root: calls}, max_depth=1)

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(self.count, int) or self.abstract_control_width is not None:
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
