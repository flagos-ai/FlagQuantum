# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/multiplication.py and its notebook.

"""The eight source multiplication examples as native register metadata."""

from collections.abc import Hashable

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_product_cost,
    source_sum_of_squares_cost,
)
from flagquantum.core.ir.program.register_types import (
    fixed_point_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature

MultiplicationExample = tuple[str, Signature, dict[Hashable, SymbolicInt]]
TOFFOLI = FixedTCall("toffoli")
T = "t"
CX = CliffordCall("cx", 2)
X = CliffordCall("x", 1)


def integer_multiplication_examples() -> tuple[MultiplicationExample, ...]:
    """Preserve original port order, fixed-point formats and declared calls.

    The fixed-point square's two input ports and the reciprocal's mixed THRU
    result are source metadata. Their executable native contracts differ.
    """
    right = RegisterSide.RIGHT
    uint = unsigned_register_type
    fxp = fixed_point_register_type
    fraction = fxp(10, 10)
    mixed = fxp(10, 7)
    examples: tuple[MultiplicationExample, ...] = (
        (
            "product_accumulation",
            Signature(
                (
                    Register("a", uint(2)),
                    Register("b", uint(2)),
                    Register("result", uint(4)),
                )
            ),
            {T: 8 * 2**2},
        ),
        (
            "integer_product",
            Signature(
                (
                    Register("a", uint(4)),
                    Register("b", uint(6)),
                    Register("result", uint(12), side=right),
                )
            ),
            {TOFFOLI: source_product_cost(4, 6).gates.toffoli},
        ),
        (
            "integer_square",
            Signature(
                (Register("a", uint(8)), Register("result", uint(16), side=right))
            ),
            {TOFFOLI: 8 * (8 - 1)},
        ),
        (
            "sum_of_squares",
            Signature(
                (
                    Register("input", uint(8), shape=(4,)),
                    Register("result", uint(18), side=right),
                )
            ),
            {TOFFOLI: source_sum_of_squares_cost(8, 4).gates.toffoli},
        ),
        (
            "fixed_point_scaling",
            Signature(
                (
                    Register("real_in", fxp(12, 12)),
                    Register("int_in", uint(4)),
                    Register("result", fxp(12, 8), side=right),
                )
            ),
            {TOFFOLI: 12 * (2 * 4 - 1) - 4**2},
        ),
        (
            "fixed_point_product",
            Signature(
                (
                    Register("a", fraction),
                    Register("b", fraction),
                    Register("result", fraction, side=right),
                )
            ),
            {TOFFOLI: 10**2 - 10 - 1},
        ),
        (
            "fixed_point_square",
            Signature(
                (
                    Register("a", fraction),
                    Register("b", fraction),
                    Register("result", fraction, side=right),
                )
            ),
            {TOFFOLI: 10**2 // 2 - 4},
        ),
        (
            "reciprocal",
            Signature((Register("a", mixed), Register("result", mixed))),
            {
                TOFFOLI: 2,
                CX: 4,
                ("multi_control_x", (1, 1, 1)): 1,
                X: 1,
                ("fixed_point_square", 10): 4,
                ("fixed_point_product", 10): 4,
                ("fixed_point_scaling", 10, 2): 4,
                ("unsigned_subtraction", 10): 4,
            },
        ),
    )
    return examples


def integer_multiplication_example_graph(
    example: MultiplicationExample,
) -> SymbolicCallGraph[Hashable]:
    """Build the same one-level resource graph without allocating execution wires."""
    name, _, calls = example
    return symbolic_call_graph(name, {name: calls}, max_depth=1)
