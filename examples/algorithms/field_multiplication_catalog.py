# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf2_multiplication factories and notebook.

"""All nine multiplication examples with native signatures and declared graphs."""

from __future__ import annotations

from collections.abc import Hashable

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.finite_field.multiplication_model import (
    CX,
    TOFFOLI,
    polynomial_multiplication_calls,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_multiplication import field_scale_matrix

MultiplicationExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def field_multiplication_examples() -> tuple[MultiplicationExample, ...]:
    """Retain concrete and unrestricted-symbolic product and both constant factories.

    Tuples identify declared shift, linear-map, polynomial and constant-scale
    children. These are example graph keys. Executable operations use the
    existing Compiler owners and ordinary append protocol.
    """
    examples: list[MultiplicationExample] = []
    for degree, suffix, accumulate in [
        (4, "accumulate", True),
        (Symbol("m"), "symbolic", False),
    ]:
        encoding = binary_field_encoding(degree)
        width = encoding.bitsize
        modulus = (
            int(encoding.gf_type.irreducible_poly) if isinstance(degree, int) else None
        )
        signature = Signature(
            (
                Register("x", encoding.register_type()),
                Register("y", encoding.register_type()),
                Register(
                    "result",
                    encoding.register_type(),
                    side=RegisterSide.THRU if accumulate else RegisterSide.RIGHT,
                ),
            )
        )
        calls: dict[Hashable, SymbolicInt] = {
            TOFFOLI: width**2,
            ("shift", degree, modulus, width, False): 1,
        }
        if accumulate:
            calls[("shift", degree, modulus, width, True)] = 1
        examples.append((f"field_product_{suffix}", signature, calls))
    encoding = binary_field_encoding([3, 1, 0])
    matrix = tuple(
        tuple(int(bit) for bit in row) for row in field_scale_matrix(encoding, 5)
    )
    for suffix in ["value", "polynomial"]:
        examples.append(
            (
                f"field_scale_{suffix}",
                Signature((Register("g", encoding.register_type()),)),
                {("linear", matrix): 1},
            )
        )
    for shift, name in [(3, "polynomial_scaled_product"), (None, "polynomial_product")]:
        target_width = 9 if shift is None else 12
        signature = Signature(
            Register(name, quantum_bit_type(), shape=(width,))
            for name, width in [("f", 5), ("g", 5), ("h", target_width)]
        )
        examples.append((name, signature, polynomial_multiplication_calls(5, shift)))
    encoding = binary_field_encoding([5, 2, 0])
    for name in [
        "field_shift_forward",
        "field_product_karatsuba",
        "field_shift_inverse",
    ]:
        if name == "field_product_karatsuba":
            signature = Signature(
                (
                    Register("x", encoding.register_type()),
                    Register("y", encoding.register_type()),
                    Register(
                        "result", encoding.register_type(), side=RegisterSide.RIGHT
                    ),
                )
            )
            calls = {
                CX: 8,
                ("polynomial", 3): 2,
                ("polynomial", 2): 1,
                ("scale", 5, 37, 9, False): 1,
                ("scale", 5, 37, 9, True): 1,
                ("shift", 5, 37, 3, False): 1,
            }
        else:
            signature = Signature((Register("f", encoding.register_type()),))
            calls = {CX: 3}
        examples.append((name, signature, calls))
    return tuple(examples)


def field_multiplication_example_graph(
    example: MultiplicationExample,
) -> SymbolicCallGraph[Hashable]:
    """Render the complete original one-level declaration, with zero gate emission."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
