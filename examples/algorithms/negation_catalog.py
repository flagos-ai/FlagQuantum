# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/negate.py.

"""The original signed negation examples as native signatures and resources."""

from collections.abc import Hashable

from sympy import Symbol

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.register_types import signed_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

NegationExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def negation_examples() -> tuple[NegationExample, ...]:
    """Retain the source BitwiseNot and AddK(1) identities at both widths."""
    examples: list[NegationExample] = []
    for name, width in (("negate_8", 8), ("negate_symbolic", Symbol("n"))):
        encodings: tuple[ArithmeticEncoding, ...] = ("twos_complement",)
        examples.append(
            (
                name,
                Signature((Register("x", signed_register_type(width)),)),
                {
                    ArithmeticCall("bitwise_not", (width,), encodings): 1,
                    ArithmeticCall("constant_add", (width,), encodings, constant=1): 1,
                },
            )
        )
    return tuple(examples)


def negation_example_graph(example: NegationExample) -> SymbolicCallGraph[Hashable]:
    """Display the original one-level resource graph without emitting gates."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
