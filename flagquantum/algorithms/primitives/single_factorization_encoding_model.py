# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 single-factorization block encodings.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sf/single_factorization.py

"""Ordered source resource models for single-factorization block encodings."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.prepare_reflection import source_prepare_reflection_cost
from .single_factorization_resource_model import (
    _controlled_swap_cost,
    _spatial_width,
    source_single_factorization_inner_prepare_cost,
    source_single_factorization_outer_prepare_cost,
    source_single_factorization_select_cost,
)

_HADAMARD = ComponentCost("Hadamard", 1, GateCost(clifford=1))


def source_single_factorization_one_body_cost(
    auxiliaries: int,
    spin_orbitals: int,
    alias_bits: int,
    rotation_bits: int = 8,
    *,
    forward_blocks: tuple[int, int] = (1, 1),
    inverse_blocks: tuple[int, int] = (1, 1),
) -> ComponentCost:
    """Preserve the ordered PREPARE, symmetry gates, SELECT and PREPARE adjoint."""
    n = _spatial_width(spin_orbitals)
    common = (auxiliaries, spin_orbitals, alias_bits, rotation_bits)
    prepare = source_single_factorization_inner_prepare_cost(*common, *forward_blocks)
    unprepare = source_single_factorization_inner_prepare_cost(
        *common, *inverse_blocks, adjoint=True
    )
    swap = _controlled_swap_cost(n)
    select = source_single_factorization_select_cost(spin_orbitals)
    # Three flags, l, p/q, rot_aa, swap_pq, spin, and the spin orbitals.
    width = 6 + auxiliaries.bit_length() + 2 * n + spin_orbitals
    return ComponentCost(
        "SingleFactorizationOneBody",
        width,
        calls=tuple(
            CostCall(leaf, repeats, width - used)
            for leaf, repeats, used in (
                (prepare, 1, prepare.qubits),
                (_HADAMARD, 2, 1),
                (swap, 1, swap.qubits),
                (select, 1, select.qubits),
                (swap, 1, swap.qubits),
                (_HADAMARD, 2, 1),
                (unprepare, 1, unprepare.qubits),
            )
        ),
    )


def source_single_factorization_block_encoding_cost(
    auxiliaries: int,
    spin_orbitals: int,
    alias_bits: int,
    inner_rotation_bits: int = 8,
    *,
    forward_blocks: tuple[int, int] = (1, 1),
    inverse_blocks: tuple[int, int] = (1, 1),
) -> ComponentCost:
    """Count the original outer PREPARE and two one-body calls around reflection.

    The locked outer amplitude-amplification precision does not affect
    its resource graph. The reflection uses only the success control
    supplied in the source decomposition.
    """
    body = source_single_factorization_one_body_cost(
        auxiliaries,
        spin_orbitals,
        alias_bits,
        inner_rotation_bits,
        forward_blocks=forward_blocks,
        inverse_blocks=inverse_blocks,
    )
    width = body.qubits + 1
    prepare = source_single_factorization_outer_prepare_cost(auxiliaries, alias_bits)
    unprepare = source_single_factorization_outer_prepare_cost(
        auxiliaries, alias_bits, adjoint=True
    )
    reflected = 2 * _spatial_width(spin_orbitals) + 2
    identity = ComponentCost("PrepareIdentity", reflected)
    reflection = source_prepare_reflection_cost(
        reflected, 0, identity, identity, controlled=True
    )
    # The reflection's model width includes its internally allocated phase target.
    return ComponentCost(
        "SingleFactorizationBlockEncoding",
        width,
        calls=tuple(
            CostCall(leaf, bystander_qubits=width - used)
            for leaf, used in (
                (prepare, prepare.qubits),
                (body, body.qubits),
                (reflection, reflected + 1),
                (body, body.qubits),
                (unprepare, unprepare.qubits),
            )
        ),
    )
