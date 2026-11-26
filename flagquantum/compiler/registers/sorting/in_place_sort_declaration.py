# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/lists/sort_in_place.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py

"""Original in-place sorting signature and model, independent of execution."""

from collections.abc import Hashable
from typing import cast

from ....core.ir.program.register_types import bounded_unsigned_register_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, ceil, log2, smax
from ..comparison.compare_exchange import (
    source_compare_exchange_cost,
    source_compare_exchange_resource_formula,
)
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def in_place_sort_signature(length: SymbolicInt, dtype: RegisterType) -> Signature:
    """Retain through xs/pi ports and the original bounded index dtype.

    Word order is big-endian. Metadata may be symbolic. Physical sorting uses
    the existing distinct-word emitter with caller-owned clean rank/work wires.
    """
    rank_width = ceil(log2(length))
    index = bounded_unsigned_register_type(rank_width, length)
    return Signature(
        (Register("xs", dtype, shape=(length,)), Register("pi", index, shape=(length,)))
    )


def source_in_place_sort_cost(length: int, width: int) -> ComponentCost:
    """Preserve the two Comparator calls and Xor, each repeated 3 length squared.

    The source root has no decomposition. Its width query takes the maximum
    of the root signature and individual child peaks, without adding bystanders.
    These are logical categories, independent of emitted sorting primitives.
    """
    if type(length) is not int or length < 1:
        raise ValueError("Sort model length must be a positive integer")
    repetitions = 3 * length**2
    compare = source_compare_exchange_cost(width)
    inverse = source_compare_exchange_cost(width, adjoint=True)
    return ComponentCost(
        "SortInPlace",
        length * (width + (length - 1).bit_length()),
        calls=(
            CostCall(compare, repetitions),
            CostCall(inverse, repetitions),
            CostCall(
                ComponentCost("Xor", 2 * width, GateCost(clifford=width)), repetitions
            ),
        ),
    )


def source_in_place_sort_resource_formula(
    length: SymbolicInt, width: SymbolicInt
) -> SymbolicResourceFormula:
    """Compose the original symbolic comparison fallback and index storage.

    Symbolic comparison counts differ from concretely bound widths in the
    source. Preserve that distinction until the caller binds its parameters.
    """
    if isinstance(length, (bool, float)) or isinstance(length, int) and length < 1:
        raise ValueError("Sort model length must be positive or symbolic")
    compare = source_compare_exchange_resource_formula(width)
    copies = 3 * length**2
    ports = length * (width + ceil(log2(length)))
    return SymbolicResourceFormula(
        and_bloq=2 * copies * compare.and_bloq,
        clifford=copies * (2 * compare.clifford + width),
        measurement=2 * copies * compare.measurement,
        logical_qubits=cast(SymbolicInt, smax(ports, compare.logical_qubits)),
        cswap=2 * copies * width,
    )


def source_in_place_sort_calls(length: int, width: int) -> dict[Hashable, SymbolicInt]:
    """Expose the existing concrete model's three direct calls for diagrams."""
    root = source_in_place_sort_cost(length, width)
    return {
        ("comparator", width): root.calls[0].repetitions,
        ("comparator_adjoint", width): root.calls[1].repetitions,
        ("xor", width): root.calls[2].repetitions,
    }
