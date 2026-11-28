# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 projectile first-quantization signatures.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py

"""Native register contracts for the projectile PREPARE and SELECT models."""

from __future__ import annotations

from ..core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt


def _bit(name: str, shape: tuple[int, ...] = ()) -> Register:
    return Register(name, quantum_bit_type(), shape=shape)


def _word(name: str, width: SymbolicInt, shape: tuple[int, ...] = ()) -> Register:
    return Register(name, opaque_register_type(width), shape=shape)


def _index(name: str, width: SymbolicInt, bound: SymbolicInt | None = None) -> Register:
    return Register(
        name,
        bounded_unsigned_register_type(width, 2**width if bound is None else bound),
    )


def projectile_prepare_tuv_signature() -> Signature:
    """Keep the term selector's four flags and two independent index words."""
    return Signature(
        (_bit("tuv"), _word("tepm", 2), _word("uv", 2), _bit("flags", (4,)))
    )


def _require_dimensions(n: int, eta: int, atoms: int, m: int) -> None:
    if any(type(value) is not int or value < 1 for value in (n, eta, atoms, m)):
        raise ValueError("First-quantized dimensions must be positive integers")


def _selection_declarations(
    n: SymbolicInt, eta: int, atoms: int, m: int, *, prepare: bool
) -> tuple[Register, ...]:
    width = (eta - 1).bit_length()
    coordinates = (
        (_index("w", 2, 3), _index("w_mean", 2, 3))
        if prepare
        else (_index("w", 3), _index("w_mean", 3))
    )
    return (
        _index("i", width, eta),
        _index("j", width, eta),
        *coordinates,
        *(_index(name, n) for name in ("r", "s", "mu")),
        *(_index(name, n + 1) for name in ("nu_x", "nu_y", "nu_z")),
        _index("m", (m - 1).bit_length()),
        _index("l", (atoms - 1).bit_length()),
    )


def projectile_prepare_signature(n: int, eta: int, atoms: int, m: int) -> Signature:
    """Describe the projectile preparation's ordered through and flag words."""
    _require_dimensions(n, eta, atoms, m)
    return _projectile_prepare_declaration(n, eta, atoms, m)


def _projectile_prepare_declaration(
    n: SymbolicInt, eta: int, atoms: int, m: int
) -> Signature:
    return Signature(
        (
            _index("tuv", 1),
            _index("tepm", 2, 3),
            _index("uv", 2),
            *_selection_declarations(n, eta, atoms, m, prepare=True),
            _bit("succ_nu"),
            _bit("plus_t"),
            _bit("flags", (4,)),
        )
    )


def projectile_select_signature(
    p: int, n: int, eta: int, atoms: int, m: int
) -> Signature:
    """Describe both system banks and the independent control and index words."""
    if type(p) is not int or p < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    _require_dimensions(n, eta, atoms, m)
    return _projectile_select_declaration(p, n, eta, atoms, m)


def _projectile_select_declaration(
    p: SymbolicInt, n: SymbolicInt, eta: int, atoms: int, m: int
) -> Signature:
    return Signature(
        (
            _bit("ham_ctrl", (4,)),
            _bit("i_ne_j"),
            _bit("plus_t"),
            *_selection_declarations(n, eta, atoms, m, prepare=False),
            _word("sys", p, (eta, 3)),
            _word("proj", n, (3,)),
        )
    )


def projectile_select_t_signature(num_bits_n: int) -> Signature:
    """Expose the two coordinate selectors and the projectile momentum bank."""
    if type(num_bits_n) is not int or num_bits_n < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    return _projectile_select_t_declaration(num_bits_n)


def _projectile_select_t_declaration(num_bits_n: SymbolicInt) -> Signature:
    return Signature(
        (
            _bit("flag_T"),
            _bit("flag_mean"),
            _bit("plus"),
            _word("w", 3),
            _word("w_mean", 3),
            _word("r", num_bits_n),
            _word("s", num_bits_n),
            _word("p", num_bits_n, (3,)),
        )
    )


def projectile_select_uv_signature(
    num_bits_p: int, num_bits_n: int, num_atoms: int, num_bits_nuc_pos: int
) -> Signature:
    """Expose independent electron, projectile, and transfer momentum widths."""
    if any(
        type(width) is not int or width < 1
        for width in (num_bits_p, num_bits_n, num_atoms, num_bits_nuc_pos)
    ):
        raise ValueError("First-quantized dimensions must be positive integers")
    return _projectile_select_uv_declaration(
        num_bits_p, num_bits_n, num_atoms, num_bits_nuc_pos
    )


def _projectile_select_uv_declaration(
    num_bits_p: SymbolicInt,
    num_bits_n: SymbolicInt,
    num_atoms: int,
    num_bits_nuc_pos: SymbolicInt,
) -> Signature:
    return Signature(
        (
            _bit("flag_tuv"),
            _bit("flag_uv"),
            _word("l", (num_atoms - 1).bit_length()),
            _word("rl", num_bits_nuc_pos),
            _word("nu", num_bits_n + 1, (3,)),
            _word("p", num_bits_n, (3,)),
            _word("q", num_bits_p, (3,)),
        )
    )
