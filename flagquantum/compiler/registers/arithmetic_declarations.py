# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic Add, CAdd, BitwiseNot and _shims.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic

# Always: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/always.py

"""Register-owned source call recipes, independent of resource query policy."""

from dataclasses import replace

from .._component_trace.arithmetic_call import ArithmeticCall, ArithmeticOperation
from .._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    FixedTCall,
    OnEachCall,
    ResourceCall,
)
from .._component_trace.symbolic_math import SymbolicInt
from .hamming_weight import hamming_weight_resource_widths


def arithmetic_callees(call: ArithmeticCall) -> dict[ResourceCall, SymbolicInt]:
    """Return direct source declarations, retaining insertion order and weights.

    Adjoint declarations adjoint each callee in place, as the original Adjoint
    wrapper does; they do not reverse an executable gate sequence. Zero-width
    BitwiseNot retains the original missing-register failure as an explicit error.
    """
    if call.adjoint:
        result: dict[ResourceCall, SymbolicInt] = {}
        for child, count in arithmetic_callees(replace(call, adjoint=False)).items():
            inverse = _adjoint(child)
            result[inverse] = result.get(inverse, 0) + count
        return result
    width = call.widths[0]
    if call.operation in ("controlled_constant_add", "controlled_xor_constant"):
        if call.control_value == 0:
            return {replace(call, control_value=1): 1, CliffordCall("x", 1): 2}
        plain_operation: ArithmeticOperation = (
            "constant_add"
            if call.operation == "controlled_constant_add"
            else "xor_constant"
        )
        plain = replace(call, operation=plain_operation)
        children = arithmetic_callees(plain)
        if plain_operation == "xor_constant":
            return {CliffordCall("cx", 2): children[CliffordCall("x", 1)]}
        return {
            (
                replace(child, operation="controlled_xor_constant")
                if isinstance(child, ArithmeticCall)
                and child.operation == "xor_constant"
                else child
            ): count
            for child, count in children.items()
        }
    if call.operation == "constant_add":
        constant = call.constant
        assert constant is not None
        if (
            type(constant) is int
            and constant < 0
            and call.encodings[0] != "twos_complement"
        ):
            constant %= 2**width
        load = ArithmeticCall(
            "xor_constant", (width,), call.encodings, constant=constant
        )
        addition = ArithmeticCall(
            "add", (width, width), call.encodings * 2, always=True
        )
        return {load: 2, addition: 1}
    if call.operation == "xor_constant":
        constant = call.constant
        assert constant is not None
        if type(width) is not int or type(constant) is not int:
            flips = width
        else:
            if call.encodings[0] == "twos_complement":
                if width < 1 or not -(1 << (width - 1)) <= constant < 1 << (width - 1):
                    raise ValueError(
                        "XOR constant is outside the source signed register range"
                    )
                pattern = constant % (1 << width)
            else:
                if not 0 <= constant < 1 << width:
                    raise ValueError(
                        "XOR constant is outside the source unsigned register range"
                    )
                pattern = constant
            flips = pattern.bit_count()
        return {CliffordCall("x", 1): flips}
    if call.operation == "hamming_weight":
        junk, _ = hamming_weight_resource_widths(width)
        return {
            ConjunctionCall((1, 1)): junk,
            CliffordCall("cx", 2): 5 * junk + (width - junk),
        }
    if call.operation == "xor":
        return {CliffordCall("cx", 2): width}
    if call.operation == "controlled_xor":
        if call.control_value == 0:
            return {replace(call, control_value=1): 1, CliffordCall("x", 1): 2}
        return {FixedTCall("toffoli"): width}
    if call.operation == "add":
        n = call.widths[1]
        return {
            ConjunctionCall((1, 1)): n - 1,
            ConjunctionCall((1, 1), adjoint=True): n - 1,
            CliffordCall("cx", 2): (n - 2) * 6 + 3,
        }
    if call.operation == "out_of_place_add":
        extra = int(call.include_most_significant_bit)
        return {
            ConjunctionCall((1, 1)): width + extra - 1,
            CliffordCall("cx", 2): 5 * (width - 1) + 2 + 3 * extra,
        }
    if call.operation == "controlled_add":
        return {
            ConjunctionCall((call.control_value, 1)): width,
            ArithmeticCall("add", call.widths, call.encodings): 1,
            ConjunctionCall((call.control_value, 1), adjoint=True): width,
        }
    if call.operation == "bitwise_not":
        if width == 0:
            raise ValueError("Original zero-width BitwiseNot has no x register")
        return {OnEachCall(CliffordCall("x", 1), width): 1}
    if call.operation == "controlled_subtraction":
        return {
            ArithmeticCall(
                "controlled_add", (width, width), ("montgomery", "montgomery")
            ): 1,
            ArithmeticCall("bitwise_not", (width,), ("montgomery",)): 3,
        }
    if call.operation == "multi_controlled_x":
        return {FixedTCall("toffoli"): width - 2}
    if call.operation == "less_than":
        return {FixedTCall("toffoli"): width}
    if call.operation == "controlled_half":
        return {FixedTCall("fredkin"): width}
    raise AssertionError("Unrecognized validated arithmetic identity")


def _adjoint(call: ResourceCall) -> ResourceCall:
    if isinstance(call, ArithmeticCall):
        return (
            call
            if call.operation
            in (
                "xor",
                "controlled_xor",
                "bitwise_not",
                "xor_constant",
                "controlled_xor_constant",
            )
            else replace(call, adjoint=not call.adjoint)
        )
    if isinstance(call, ConjunctionCall):
        return replace(call, adjoint=not call.adjoint)
    if isinstance(call, (CliffordCall, FixedTCall, OnEachCall)):
        # These recipes only contain self-adjoint X/CNOT/Toffoli/Fredkin.
        return call
    raise AssertionError("Arithmetic recipe contains an unsupported adjoint")


def arithmetic_call_graph(
    root: ArithmeticCall,
) -> dict[ResourceCall, dict[ResourceCall, SymbolicInt]]:
    """Expand arithmetic and OnEach nodes, stopping at logical QEC primitives.

    Return the existing declaration-map format. Query policy and optional cost
    replacements remain owned by generic component-trace consumers.
    """
    declarations: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {}

    def visit(node: ResourceCall) -> None:
        if node in declarations:
            return
        if isinstance(node, ArithmeticCall):
            children = arithmetic_callees(node)
        elif isinstance(node, OnEachCall):
            children = {node.child: node.width}
        else:
            return
        declarations[node] = children
        for child in children:
            visit(child)

    visit(root)
    return declarations
