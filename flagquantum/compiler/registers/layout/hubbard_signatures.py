# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Port declarations follow Qualtran v0.7.0 select_hubbard.py.

"""Declare square-lattice Hubbard words without owning numerical algorithms."""

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, ceil, log2


def _selection_ports(coordinate: RegisterType) -> tuple[Register, ...]:
    flag = bounded_unsigned_register_type(1, 2)
    return (
        Register("U", flag),
        Register("V", flag),
        Register("p_x", coordinate),
        Register("p_y", coordinate),
        Register("alpha", flag),
        Register("q_x", coordinate),
        Register("q_y", coordinate),
        Register("beta", flag),
    )


def hubbard_preparation_signature(dimension: int) -> Signature:
    """Keep PREPARE's exact integer widths and caller-owned two-bit temp word."""
    coordinate = bounded_unsigned_register_type((dimension - 1).bit_length(), dimension)
    return Signature(
        (*_selection_ports(coordinate), Register("temp", opaque_register_type(2)))
    )


def hubbard_signature(
    kind: str, dimension: SymbolicInt, *, controlled: bool = False
) -> Signature:
    """Keep source control, bounded coordinate and opaque target port order."""
    coordinate = bounded_unsigned_register_type(ceil(log2(dimension)), dimension)
    flag = bounded_unsigned_register_type(1, 2)
    ports: tuple[Register, ...]
    if kind == "majorana":
        ports = (
            Register("x", coordinate),
            Register("y", coordinate),
            Register("spin", flag),
        )
    elif kind == "spin_up":
        ports = (
            Register("V", flag),
            Register("x", coordinate),
            Register("y", coordinate),
        )
    elif kind == "select":
        ports = _selection_ports(coordinate)
    else:
        raise ValueError("Unknown Hubbard port family")
    return Signature(
        (
            *((Register("control", quantum_bit_type()),) if controlled else ()),
            *ports,
            Register("target", opaque_register_type(2 * dimension**2)),
        )
    )
