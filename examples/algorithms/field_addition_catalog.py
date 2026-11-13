# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from frozen Qualtran v0.7.0 gf2_addition and gf2_add_k notebooks.

"""Native finite-field addition metadata and one-level resource graph examples."""

from __future__ import annotations

from collections.abc import Sequence

from sympy import Symbol, symbols

from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.finite_field.addition_model import (
    field_add_constant_calls,
    field_addition_calls,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.call_graph import render_call_graph_svg
from flagquantum.drawer.register_graph_table import signature_instance_table
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.simulation.finite_field_addition import binary_field_encoding

FieldExample = tuple[str, Signature, dict[CliffordCall, SymbolicInt]]


def field_addition_examples(*, constant: bool = False) -> tuple[FieldExample, ...]:
    """Preserve all four source examples, symbol assumptions and register order."""
    m, k = (
        symbols("m, k", positive=True, integers=True)
        if constant
        else (Symbol("m"), None)
    )
    examples = []
    for width, value, suffix in [(4, 1, "concrete"), (m, k, "symbolic")]:
        dtype = binary_field_encoding(width).register_type()
        signature = Signature(
            tuple(
                Register(name, dtype) for name in (("x",) if constant else ("x", "y"))
            )
        )
        calls = (
            field_add_constant_calls(width, value)
            if constant
            else field_addition_calls(width)
        )
        examples.append(
            (f"field_add{'_constant' if constant else ''}_{suffix}", signature, calls)
        )
    return tuple(examples)


def field_signature_svg(examples: Sequence[FieldExample]) -> bytes:
    """Show symbolic and concrete field signatures using native table layout."""
    nodes = []
    for index, (name, signature, _) in enumerate(examples):
        names = tuple(reg.name for reg in signature)
        step = RegisterStep(name, names, names)
        ports = {reg.name: f"p{i}" for i, reg in enumerate(signature)}
        table = signature_instance_table(signature, step, ports)
        nodes.append(f"n{index} [shape=plain, label=<{table}>];")
    return render_call_graph_svg("digraph signatures { " + " ".join(nodes) + " }")
