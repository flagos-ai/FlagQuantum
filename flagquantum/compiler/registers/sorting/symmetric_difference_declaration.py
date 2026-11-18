# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/lists/symmetric_difference.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/symmetric_difference.py

"""Original set-difference metadata and declared costs, without compaction."""

from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt
from ..bits.hamming_weight_resource_model import source_hamming_weight_cost
from ..comparison.equality import source_equality_constant_cost, source_equality_cost
from .bitonic_merge import source_bitonic_merge_cost


def symmetric_difference_signature(
    left_size: SymbolicInt,
    right_size: SymbolicInt,
    output_size: SymbolicInt,
    dtype: RegisterType,
) -> Signature:
    """Preserve all four through ports, dtype and symbolic array dimensions."""
    if isinstance(left_size, int) and isinstance(right_size, int):
        assert left_size >= right_size, "lhs must be the larger set"
    return Signature(
        (
            Register("S", dtype, shape=(left_size,)),
            Register("T", dtype, shape=(right_size,)),
            Register("diff", dtype, shape=(output_size,)),
            Register("flag", quantum_bit_type()),
        )
    )


def source_symmetric_difference_cost(
    left_size: int, right_size: int, output_size: int, width: int
) -> ComponentCost:
    """Compose the actual source graph and its individual-child width fallback.

    The source merge uses left_size as both half lengths, including unequal
    input sets. Its floor cardinality predicate and unconditional word copies
    remain model data. Executable parity checking and compaction are separate.
    """
    if (
        any(type(n) is not int for n in (left_size, right_size, output_size, width))
        or not left_size >= right_size >= 1
        or output_size < 0
        or width < 1
    ):
        raise ValueError(
            "Set model requires left >= right >= 1, nonnegative output and positive width"
        )
    adjacent = left_size + right_size - 1
    constant_width = adjacent.bit_length()
    constant = (left_size + right_size - output_size) // 2
    merge = source_bitonic_merge_cost(left_size, width)
    equal = source_equality_cost(width)
    population = source_hamming_weight_cost(adjacent)
    predicate = source_equality_constant_cost(constant_width, constant)
    # Counter insertion order is copies, flag, then each forward/adjoint pair.
    calls = (
        CostCall(
            ComponentCost("Xor", 2 * width, GateCost(clifford=width)), output_size
        ),
        CostCall(ComponentCost("CNOT", 2, GateCost(clifford=1))),
        CostCall(merge),
        CostCall(source_bitonic_merge_cost(left_size, width, adjoint=True)),
        CostCall(equal, adjacent),
        CostCall(
            ComponentCost("EqualsAdjoint", equal.qubits, equal.gates, equal.calls),
            adjacent,
        ),
        CostCall(population),
        CostCall(source_hamming_weight_cost(adjacent, adjoint=True)),
        CostCall(predicate),
        CostCall(
            ComponentCost(
                "EqualsAConstantAdjoint",
                predicate.qubits,
                predicate.gates,
                predicate.calls,
            )
        ),
    )
    return ComponentCost(
        "SymmetricDifference",
        (left_size + right_size + output_size) * width + 1,
        calls=calls,
    )
