# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/subtraction.py.

"""Native signatures and resource views for the original subtraction examples."""

from collections import Counter
from collections.abc import Hashable

from sympy import Symbol

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    OnEachCall,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.core.ir.program.register_types import signed_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

SubtractionExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]


def symbolic_subtraction_steps(
    width: SymbolicInt,
) -> tuple[tuple[Hashable, tuple[str, ...]], ...]:
    """Keep the equal-width signed composite's logical operation and port order."""
    to_unsigned = ("cast", "signed", "unsigned", width)
    to_signed = ("cast", "unsigned", "signed", width)
    complement = OnEachCall(CliffordCall("x", 1), width)
    return (
        (to_unsigned, ("a",)),
        (to_unsigned, ("b",)),
        (complement, ("a",)),
        (ArithmeticCall("add", (width, width)), ("a", "b")),
        (complement, ("b",)),
        (complement, ("a",)),
        (to_signed, ("b",)),
        (to_signed, ("a",)),
    )


def subtraction_examples() -> tuple[SubtractionExample, ...]:
    """Reproduce every source factory with neutral metadata and declared children."""
    n = Symbol("n")
    examples: list[SubtractionExample] = []
    for name, left, right, direction in (
        ("subtract_symbolic", n, n, "subtract"),
        ("subtract_small", 4, 4, "subtract"),
        ("subtract_large", 64, 64, "subtract"),
        ("subtract_unequal", 4, 16, "subtract"),
        ("subtract_symbolic_composite", n, n, "composite"),
        ("subtract_from_symbolic", n, n, "from"),
        ("subtract_from_small", 4, 4, "from"),
        ("subtract_from_large", 64, 64, "from"),
    ):
        signature = Signature(
            (
                Register("a", signed_register_type(left)),
                Register("b", signed_register_type(right)),
            )
        )
        calls: dict[Hashable, SymbolicInt]
        if direction == "from":
            calls = {
                ArithmeticCall("bitwise_not", (right,), ("twos_complement",)): 2,
                ArithmeticCall("add", (right, right), ("twos_complement",) * 2): 1,
            }
        elif direction == "composite":
            calls = dict(
                Counter(child for child, _ in symbolic_subtraction_steps(right))
            )
        else:
            calls = {
                OnEachCall(CliffordCall("x", 1), right): 3,
                ArithmeticCall("add", (right, right)): 1,
            }
            delta = right - left
            if delta:
                calls.update(
                    {
                        ("allocate", delta): 1,
                        ("free", delta): 1,
                        ("multi_target_cx", delta): 2,
                    }
                )
        examples.append((name, signature, calls))
    return tuple(examples)


def subtraction_example_graph(
    example: SubtractionExample,
) -> SymbolicCallGraph[Hashable]:
    """Render a one-level declaration without allocating any wires."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
