# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register roles follow locked Qualtran v0.7.0 chemistry/thc/prepare.py.

"""Own the named THC PREPARE ports and dataset-major retained banks."""

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..layout.partition import RegisterLayout

THC_PREPARE_TARGETS = ("theta", "alt_theta", "alt_mu", "alt_nu", "keep")


def thc_prepare_selection_registers(
    num_mu: int, precision: int
) -> tuple[Register, ...]:
    orbit = bounded_unsigned_register_type(num_mu.bit_length(), num_mu + 1)
    bit = bounded_unsigned_register_type(1)
    return (
        Register("mu", orbit),
        Register("nu", orbit),
        *(Register(name, bit) for name in ("plus_mn", "plus_a", "plus_b")),
        Register("sigma", bounded_unsigned_register_type(precision)),
        *(Register(name, bit) for name in ("rot", "succ", "nu_eq_mp1")),
    )


def thc_prepare_target_registers(layout: RegisterLayout) -> tuple[Register, ...]:
    return tuple(
        Register(
            name,
            (
                quantum_bit_type()
                if name in ("theta", "alt_theta")
                else opaque_register_type(width)
            ),
            side=RegisterSide.RIGHT,
        )
        for name, width in layout
    )


def thc_prepare_retained_registers(
    targets: tuple[Register, ...], log_block_size: int
) -> tuple[Register, ...]:
    # The original declares these arrays even when their length is zero.
    return tuple(
        Register(
            "junk_" + reg.name,
            reg.dtype,
            (*reg.shape, (1 << log_block_size) - 1),
            RegisterSide.RIGHT,
        )
        for reg in targets
    )


def thc_prepare_through_junk(address_width: int) -> tuple[Register, ...]:
    return (
        Register("s", opaque_register_type(address_width)),
        Register("less_than", quantum_bit_type()),
        Register("extra_ctrl", quantum_bit_type()),
    )


def thc_preparation_signature(
    num_mu: int, num_spin_orb: int, precision: int, log_block_size: int
) -> Signature:
    width = num_mu.bit_length()
    rows = num_mu * (num_mu + 1) // 2 + num_spin_orb // 2
    targets = thc_prepare_target_registers(
        tuple(zip(THC_PREPARE_TARGETS, (1, 1, width, width, precision), strict=True))
    )
    return Signature(
        (
            *thc_prepare_selection_registers(num_mu, precision),
            *thc_prepare_through_junk((rows - 1).bit_length()),
            *targets,
            *thc_prepare_retained_registers(targets, log_block_size),
        )
    )
