# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 GF2Inverse examples and notebook.

"""Native inverse signatures, original one-level graphs and independent models."""

from __future__ import annotations

from collections.abc import Hashable

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.finite_field.inverse_model import (
    inverse_gate_model,
    inverse_junk_register_count,
    inverse_source_calls,
)
from flagquantum.compiler.registers.finite_field.linear_resource_model import (
    source_binary_linear_resource_formula,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_frobenius import field_frobenius_matrix

InverseExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def field_inverse_example(degree: SymbolicInt) -> InverseExample:
    """Retain the complete original result/junk signature and declared children.

    The source factory uses a positive integer symbolic m. The notebook uses
    an unrestricted symbolic m. Pass the corresponding symbol unchanged.
    """
    encoding = binary_field_encoding(degree)
    registers = [
        Register("x", encoding.register_type()),
        Register("result", encoding.register_type(), side=RegisterSide.RIGHT),
    ]
    if is_symbolic_value(degree) or degree > 1:
        registers.append(
            Register(
                "junk",
                encoding.register_type(),
                shape=(inverse_junk_register_count(degree),),
                side=RegisterSide.RIGHT,
            )
        )
    name = (
        "field_inverse_symbolic"
        if is_symbolic_value(degree)
        else f"field_inverse_degree_{degree}"
    )
    return name, Signature(registers), inverse_source_calls(degree)


def field_inverse_example_graph(example: InverseExample) -> SymbolicCallGraph[Hashable]:
    """Keep the original complete one-level inverse declaration and leaf counts."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)


def field_inverse_example_gate_model(degree: int) -> dict[Hashable, SymbolicInt]:
    """Evaluate exact concrete source counts through independent child owners."""
    encoding = binary_field_encoding(degree)
    powers = {
        child[2]
        for child in inverse_source_calls(degree)
        if isinstance(child, tuple) and child[0] == "square"
    }
    costs = {}
    for power in powers:
        matrix = field_frobenius_matrix(encoding, power)
        costs[power] = source_binary_linear_resource_formula(
            tuple(tuple(int(b) for b in row) for row in matrix)
        ).clifford
    return inverse_gate_model(
        degree, modulus=int(encoding.gf_type.irreducible_poly), square_cliffords=costs
    )
