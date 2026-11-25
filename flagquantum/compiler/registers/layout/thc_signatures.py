# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Signatures follow locked Qualtran v0.7.0 chemistry/thc/select_bloq.py.

"""Literal THC ports, independent of native numerical parameter admission."""

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def thc_rotation_signature(
    num_mu: int, num_spin_orb: SymbolicInt, angle_width: SymbolicInt
) -> Signature:
    """Declare the model-only leaf's flag, shared angle word, address and bank."""
    return Signature(
        (
            Register("nu_eq_mp1", quantum_bit_type()),
            Register("data", opaque_register_type(angle_width)),
            Register("sel", opaque_register_type(num_mu.bit_length())),
            Register("trg", opaque_register_type(num_spin_orb // 2)),
        )
    )


def thc_selection_signature(
    num_mu: int,
    num_spin_orb: SymbolicInt,
    keep_width: SymbolicInt,
    *,
    control_value: int | None = None,
) -> Signature:
    """Retain the original selection order and optional pass-through control."""
    bit = bounded_unsigned_register_type(1)
    orbital = bounded_unsigned_register_type(num_mu.bit_length(), num_mu + 1)
    system = opaque_register_type(num_spin_orb // 2)
    return Signature(
        (
            *(
                (Register("control", quantum_bit_type()),)
                if control_value is not None
                else ()
            ),
            Register("succ", bit),
            Register("nu_eq_mp1", bit),
            Register("mu", orbital),
            Register("nu", orbital),
            *(Register(name, bit) for name in ("plus_mn", "plus_a", "plus_b")),
            Register("sigma", bounded_unsigned_register_type(keep_width)),
            Register("rot", bit),
            Register("sys_a", system),
            Register("sys_b", system),
        )
    )
