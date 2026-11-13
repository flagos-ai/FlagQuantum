# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 GF2Inverse properties and call graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_inverse.py

"""Independent Itoh--Tsujii declarations and static signature resource models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from ....core.value_parameters import is_symbolic_value
from ..._component_trace.symbolic_math import SymbolicInt, bit_length
from .multiplication_model import CX, TOFFOLI, field_multiplication_gate_model


def inverse_junk_register_count(width: SymbolicInt) -> SymbolicInt:
    """Retain the original junk-shape formula, even when degree one omits the port."""
    length = bit_length(width - 1)
    if is_symbolic_value(width):
        return 2 * length - 2
    return length - 2 + max(int(width - 1).bit_count() - 1, 1)


def inverse_source_calls(width: SymbolicInt) -> dict[Hashable, SymbolicInt]:
    """Retain complete concrete child identities and the symbolic shape estimate.

    Tuple keys identify field addition, multiplication, square/count/direction
    and a shape-only binary linear map. No executable gates are constructed.
    """
    if width == 1:
        return {("addition", width): 1}
    top = bit_length(width - 1) - 1
    if is_symbolic_value(width):
        return {
            ("addition", width): 2 * top,
            ("product", width): top + bit_length(width - 1) - 1,
            ("linear_bound", width): 2 * top + bit_length(width - 1),
        }
    hamming = int(width - 1).bit_count()
    calls: dict[Hashable, SymbolicInt] = {
        ("square", width, 2 ** (i - 1), False): 1 for i in range(2, top + 1)
    }
    calls.update({("square", width, 2 ** (i - 1), True): 1 for i in range(1, top + 1)})
    calls[("square", width, 1, False)] = 1 + (top > 0)
    bits = [i for i in reversed(range(top + 1)) if int(width - 1) >> i & 1]
    for bit in bits[1:]:
        child = ("square", width, 2**bit, False)
        calls[child] = calls.get(child, 0) + 1
    multiplications = top + hamming - 1
    if multiplications:
        calls[("product", width)] = multiplications
    additions = 2 * top + (width == 2)
    if additions:
        calls[("addition", width)] = additions
    return calls


def inverse_gate_model(
    width: SymbolicInt,
    *,
    modulus: int | None = None,
    square_cliffords: Mapping[int, SymbolicInt] | None = None,
) -> dict[Hashable, SymbolicInt]:
    """Combine source declarations and independent child costs.

    Concrete square costs are supplied by the numerical matrix/PLU owner.
    Symbolic fields retain the original shape-only bound. This reports raw
    compute counts. A clean XOR wrapper costs twice the raw gates plus width CX.
    Static output width and actual allocated peak are separate accounts.
    """
    calls = inverse_source_calls(width)
    result: dict[Hashable, SymbolicInt] = {CX: 0, TOFFOLI: 0}
    for child, count in calls.items():
        if not isinstance(child, tuple):
            raise TypeError("Inverse declarations require tuple child identities")
        if child[0] == "addition":
            result[CX] += count * width
        elif child[0] == "product":
            model = field_multiplication_gate_model(
                bit_length(2**width - 1), modulus=modulus, algorithm="karatsuba"
            )
            result[CX] += count * model.get(CX, 0)
            result[TOFFOLI] += count * model.get(TOFFOLI, 0)
        elif child[0] == "linear_bound":
            result[CX] += count * width * (width - 1)
        else:
            if square_cliffords is None:
                raise ValueError(
                    "Concrete inverse models require independent square costs"
                )
            result[CX] += count * square_cliffords[child[2]]
    return result
