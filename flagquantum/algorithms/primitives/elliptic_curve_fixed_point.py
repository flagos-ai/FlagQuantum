# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECAddR.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_add_r.py

"""Fixed-point affine addition and source port metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from .elliptic_curve_point import ECPoint


def fixed_point_addition_signature(
    width: SymbolicInt, window: SymbolicInt | None = None
) -> Signature:
    """Declare source unsigned coordinates and scalar or shaped control."""
    word = unsigned_register_type(width)
    control_shape = () if window is None else (window,)
    return Signature(
        (
            Register("ctrl", quantum_bit_type(), shape=control_shape),
            Register("x", word),
            Register("y", word),
        )
    )


@dataclass(frozen=True)
class EllipticCurveFixedPointAddition:
    """Source fixed-point classical map on raw affine coordinates."""

    width: SymbolicInt
    point: ECPoint

    @property
    def source_signature(self) -> Signature:
        return fixed_point_addition_signature(self.width)

    def classical_values(
        self, ctrl: int, x: SymbolicInt, y: SymbolicInt
    ) -> tuple[int, SymbolicInt, SymbolicInt]:
        if ctrl == 0:
            return ctrl, x, y
        current = ECPoint(x, y, self.point.mod, self.point.curve_a)
        output = current + self.point
        return 1, output.x, output.y
