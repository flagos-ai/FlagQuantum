# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 gf_poly_arithmetic examples and notebooks.

"""Complete polynomial-addition and gate-free coefficient-view examples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable

from sympy import Symbol, symbols

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.finite_field.addition_model import (
    field_add_constant_calls,
    field_addition_calls,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

PolynomialExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def polynomial_addition_examples(
    *, constant: bool = False
) -> tuple[PolynomialExample, ...]:
    """Retain both original factories, including constant zero in the symbolic one.

    Child tuples hold field width and optional constant. Strings denote the
    two polynomial view leaves. Concrete constant addition includes these
    leaves because the original graph delegates to its full decomposition.
    """
    n, m = symbols("n, m", positive=True, integers=True)
    examples: list[PolynomialExample] = []
    for degree, width, suffix in [(4, 3, "concrete"), (n, m, "symbolic")]:
        encoding = FiniteFieldPolynomialEncoding(degree, binary_field_encoding(width))
        coefficient_width = encoding.field.bitsize
        signature = Signature(
            Register(name, encoding.register_type())
            for name in (("f_x",) if constant else ("f_x", "g_x"))
        )
        calls: dict[Hashable, SymbolicInt]
        if not constant:
            calls = {(coefficient_width, None): degree + 1}
        elif suffix == "symbolic":
            # A fresh unknown coefficient is retained even for the zero polynomial.
            calls = {(coefficient_width, Symbol("_g_x0")): degree + 1}
        else:
            calls = {"Polynomial split": 1}
            calls.update(
                Counter((coefficient_width, value) for value in (1, 2, 3, 4, 5))
            )
            calls["Polynomial join"] = 1
        examples.append(
            (
                f"polynomial_add{'_constant' if constant else ''}_{suffix}",
                signature,
                calls,
            )
        )
    return tuple(examples)


def polynomial_example_graph(
    example: PolynomialExample, *, max_depth: int | None = 1
) -> SymbolicCallGraph[Hashable]:
    """Expand coefficient calls through existing field declarations, without gates."""
    root, _, calls = example
    declarations: dict[Hashable, dict[Hashable, SymbolicInt]] = {root: calls}
    for child in calls:
        if isinstance(child, tuple):
            width, value = child
            field_calls = (
                field_addition_calls(width)
                if value is None
                else field_add_constant_calls(width, value)
            )
            declarations[child] = {key: count for key, count in field_calls.items()}
    return symbolic_call_graph(root, declarations, max_depth=max_depth)


def polynomial_view_examples() -> tuple[tuple[str, Signature], ...]:
    """Preserve both degree-four GF(8) factories and opposite view directions."""
    encoding = FiniteFieldPolynomialEncoding(4, binary_field_encoding(3))
    result = []
    for name, side in [
        ("polynomial_split", RegisterSide.LEFT),
        ("polynomial_join", RegisterSide.RIGHT),
    ]:
        registers: tuple[Register, ...] = (
            Register("reg", encoding.register_type(), side=side),
            Register(
                "reg",
                encoding.field.register_type(),
                shape=(5,),
                side=RegisterSide.THRU ^ side,
            ),
        )
        if side == RegisterSide.RIGHT:
            registers = tuple(reversed(registers))
        result.append((name, Signature(registers)))
    return tuple(result)
