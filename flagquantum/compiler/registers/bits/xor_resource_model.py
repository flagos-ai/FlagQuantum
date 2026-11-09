# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned Qualtran v0.7.0 Xor, Controlled and QubitCount.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Native register XOR declarations and concrete/symbolic resource queries."""

from dataclasses import fields
from typing import cast

from ..._component_trace.arithmetic_call import ArithmeticCall, ArithmeticEncoding
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_generalization import ControlPattern
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..arithmetic.arithmetic_declarations import arithmetic_call_graph
from ..control.control_predicate import ControlRegister
from ..control.controlled_component_resource_model import (
    controlled_component_resource_formula,
    controlled_pattern_resource_formula,
)
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def source_xor_resource_formula(
    width: SymbolicInt,
    *,
    encoding: ArithmeticEncoding = "unsigned",
    control_value: int | None = None,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Query unchanged XOR declarations, preserving unknown-width fallback.

    Zero width retains the original failed QubitCount boundary. An estimate
    for symbolic width does not authorize emitting a gate sequence for it.
    """
    if type(legacy_shims) is not bool:
        raise ValueError("XOR formula profile must be boolean")
    if control_value is not None and (
        type(control_value) is not int or control_value not in (0, 1)
    ):
        raise ValueError("XOR control activation must be zero or one")
    root = ArithmeticCall(
        "xor" if control_value is None else "controlled_xor",
        (width,),
        (encoding,),
        control_value=1 if control_value is None else control_value,
    )
    if width == 0:
        raise ValueError("Original zero-width XOR has no successful qubit-count query")
    gates = query_qec_gate_cost(
        root, declared_calls=arithmetic_call_graph(root), legacy_shims=legacy_shims
    )
    peak = cast(
        SymbolicInt,
        smax(2, 2 * width) if control_value is None else smax(3, 2 * width + 1),
    )
    return SymbolicResourceFormula(
        logical_qubits=peak, **{f.name: getattr(gates, f.name) for f in fields(gates)}
    )


def source_grouped_xor_resource_formula(
    width: SymbolicInt,
    registers: tuple[ControlRegister, ...],
    *,
    encoding: ArithmeticEncoding = "unsigned",
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Keep both logical data words live around the shared control reduction."""
    positive = ArithmeticCall("controlled_xor", (width,), (encoding,))
    body = source_xor_resource_formula(
        width, encoding=encoding, control_value=1, legacy_shims=legacy_shims
    )
    return controlled_component_resource_formula(
        positive, body, registers, target_width=2 * width, legacy_shims=legacy_shims
    )


def source_pattern_controlled_xor_resource_formula(
    width: SymbolicInt,
    controls: ControlPattern,
    *,
    encoding: ArithmeticEncoding = "unsigned",
    legacy_shims: bool = False,
    control_shapes: tuple[tuple[SymbolicInt, ...], ...] | None = None,
) -> SymbolicResourceFormula:
    """Query known activations or an unknown-shaped control count around XOR."""
    positive = ArithmeticCall("controlled_xor", (width,), (encoding,))
    body = source_xor_resource_formula(
        width, encoding=encoding, control_value=1, legacy_shims=legacy_shims
    )
    return controlled_pattern_resource_formula(
        positive,
        body,
        controls,
        target_width=2 * width,
        legacy_shims=legacy_shims,
        control_shapes=control_shapes,
    )
