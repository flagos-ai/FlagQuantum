# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register roles follow locked Qualtran v0.7.0 chemistry/sparse/prepare.py.

"""Own named sparse PREPARE words and their directional bank layout."""

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..layout.partition import RegisterLayout

SPARSE_PREPARE_TARGETS = (
    "p",
    "q",
    "r",
    "s",
    "theta",
    "flag_1b",
    "alt_p",
    "alt_q",
    "alt_r",
    "alt_s",
    "alt_theta",
    "alt_flag_1b",
    "keep",
)


def sparse_prepare_selection_registers(
    rows: int, mu: int, address_width: int
) -> tuple[Register, ...]:
    bit = bounded_unsigned_register_type(1)
    return (
        Register("d", bounded_unsigned_register_type(address_width, rows)),
        Register("sigma", bounded_unsigned_register_type(mu)),
        *(
            Register(name, bit)
            for name in ("alpha", "beta", "rot_aa", "swap_pq", "swap_rs", "swap_pqrs")
        ),
    )


def sparse_prepare_target_registers(layout: RegisterLayout) -> tuple[Register, ...]:
    return tuple(
        Register(
            name,
            (
                unsigned_register_type(width)
                if name in ("p", "q", "r", "s")
                else (
                    quantum_bit_type()
                    if name in ("theta", "flag_1b", "alt_theta", "alt_flag_1b")
                    else opaque_register_type(width)
                )
            ),
            side=RegisterSide.RIGHT,
        )
        for name, width in layout
    )


def sparse_prepare_retained_registers(
    targets: tuple[Register, ...], log_block_size: int
) -> tuple[Register, ...]:
    count = (1 << log_block_size) - 1
    return (
        tuple(
            Register(
                "jnk_" + reg.name,
                reg.dtype,
                shape=(*reg.shape, count),
                side=RegisterSide.RIGHT,
            )
            for reg in targets
        )
        if count
        else ()
    )


def sparse_preparation_signature(
    num_spin_orb: int, rows: int, mu: int, log_block_size: int
) -> Signature:
    orbital = (num_spin_orb // 2 - 1).bit_length()
    widths = (orbital,) * 4 + (1,) * 2 + (orbital,) * 4 + (1,) * 2 + (mu,)
    targets = sparse_prepare_target_registers(
        tuple(zip(SPARSE_PREPARE_TARGETS, widths, strict=True))
    )
    return Signature(
        (
            *sparse_prepare_selection_registers(rows, mu, (rows - 1).bit_length()),
            Register("less_than", quantum_bit_type()),
            *targets,
            *sparse_prepare_retained_registers(targets, log_block_size),
        )
    )
