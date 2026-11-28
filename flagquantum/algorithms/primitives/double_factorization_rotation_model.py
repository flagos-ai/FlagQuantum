# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ProgRotGateArray.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/df/select_bloq.py

"""Declared basis-rotation cost for double-factorized chemistry."""

from dataclasses import dataclass

from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from .chemistry_qroam_model import source_chemistry_qroam_cost
from .double_factorization_prepare_model import (
    _TOFFOLI,
    _df_address_widths,
    _require_nonnegative_precision,
)


def source_double_factorization_programmed_rotation_signature(
    spin_orbitals: int, eigenvalues: int, rotation_bits: int
) -> Signature:
    """Preserve the explicit opaque words and pair of spin-system words."""
    _df_address_widths(spin_orbitals, eigenvalues)
    _require_nonnegative_precision(rotation_bits)
    return _double_factorization_rotation_declaration(
        spin_orbitals, eigenvalues, rotation_bits
    )


def _double_factorization_rotation_declaration(
    spin_orbitals: int, eigenvalues: int, rotation_bits: SymbolicInt
) -> Signature:
    spatial, rank, offset = _df_address_widths(spin_orbitals, eigenvalues)
    return Signature(
        (
            Register("offset", opaque_register_type(offset)),
            Register("p", opaque_register_type(rank)),
            Register("rotations", opaque_register_type(spatial * rotation_bits)),
            Register("spin", quantum_bit_type()),
            Register("sys", opaque_register_type(spatial), shape=(2,)),
        )
    )


@dataclass(frozen=True)
class ProgrammedRotationDeclaration:
    """The raw source terms before the estimator admits gate counts."""

    signature: Signature
    local_toffolis: int
    lookup_rows: int
    lookup_width: int


def source_double_factorization_programmed_rotation_declaration(
    spin_orbitals: int, eigenvalues: int, rotation_bits: int
) -> ProgrammedRotationDeclaration:
    """Retain the source signed local count, including low precision."""
    signature = source_double_factorization_programmed_rotation_signature(
        spin_orbitals, eigenvalues, rotation_bits
    )
    spatial, _, offset = _df_address_widths(spin_orbitals, eigenvalues)
    return ProgrammedRotationDeclaration(
        signature,
        offset - 1 + spin_orbitals * (rotation_bits - 2),
        eigenvalues + spatial,
        spatial * rotation_bits,
    )


def source_double_factorization_programmed_rotation_cost(
    spin_orbitals: int,
    eigenvalues: int,
    rotation_bits: int,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    """Count the source offset and rotations with the directional QROAM."""
    declaration = source_double_factorization_programmed_rotation_declaration(
        spin_orbitals, eigenvalues, rotation_bits
    )
    if declaration.local_toffolis < 0:
        raise ValueError("The locked rotation formula gives a negative gate count")
    return ComponentCost(
        "Adjoint" if adjoint else "ProgRotGateArray",
        int(declaration.signature.n_qubits()),
        calls=(
            CostCall(_TOFFOLI, declaration.local_toffolis),
            CostCall(
                source_chemistry_qroam_cost(
                    declaration.lookup_rows, declaration.lookup_width, adjoint=adjoint
                )
            ),
        ),
    )
