# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 double-factorization block encodings.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/df/double_factorization.py

"""Ordered resource models for the original double-factorization encodings."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.prepare_reflection import source_prepare_reflection_cost
from .double_factorization_prepare_model import (
    _auxiliary_width,
    _df_address_widths,
    source_double_factorization_indexed_data_cost,
    source_double_factorization_inner_prepare_cost,
    source_double_factorization_outer_prepare_cost,
)
from .double_factorization_rotation_model import (
    source_double_factorization_programmed_rotation_cost,
)

_HADAMARD = ComponentCost("Hadamard", 1, GateCost(clifford=1))


def source_double_factorization_one_body_cost(
    spin_orbitals: int,
    eigenvalues: int,
    alias_bits: int = 8,
    basis_rotation_bits: int = 24,
    inner_rotation_bits: int = 8,
) -> ComponentCost:
    """Keep the decomposition order and live widths of the locked one-body block."""
    spatial, rank, offset = _df_address_widths(spin_orbitals, eigenvalues)
    preparation = (spin_orbitals, eigenvalues, inner_rotation_bits, alias_bits)
    rotation = (spin_orbitals, eigenvalues, basis_rotation_bits)
    prep = source_double_factorization_inner_prepare_cost(*preparation)
    erase = source_double_factorization_inner_prepare_cost(*preparation, adjoint=True)
    rotate = source_double_factorization_programmed_rotation_cost(*rotation)
    unrotate = source_double_factorization_programmed_rotation_cost(
        *rotation, adjoint=True
    )
    swap = ComponentCost("CSwap", spin_orbitals + 1, GateCost(cswap=spatial))
    # The source's gate-count declaration fixes bitsize=5. Its circuit binds
    # the actual spatial width, with one additional qubit for the CCZ predicate.
    select_ports = spatial + 2
    select = ComponentCost("ApplyControlledZs", select_ports + 1, GateCost(toffoli=1))
    width = (
        5
        + 2 * rank
        + offset
        + inner_rotation_bits
        + spatial * basis_rotation_bits
        + spin_orbitals
    )
    return ComponentCost(
        "DoubleFactorizationOneBody",
        width,
        calls=tuple(
            CostCall(child, bystander_qubits=width - ports)
            for child, ports in (
                (prep, prep.qubits),
                (_HADAMARD, 1),
                (swap, swap.qubits),
                (rotate, rotate.qubits),
                (select, select_ports),
                (unrotate, unrotate.qubits),
                (swap, swap.qubits),
                (erase, erase.qubits),
                (_HADAMARD, 1),
            )
        ),
    )


def source_double_factorization_block_encoding_cost(
    auxiliaries: int,
    spin_orbitals: int,
    eigenvalues: int,
    alias_bits: int = 8,
    basis_rotation_bits: int = 24,
    inner_rotation_bits: int = 8,
) -> ComponentCost:
    """Count the locked outer preparation and repeated one-body stages.

    The original outer rotation precision has no effect on its cost graph.
    The middle reflection marks p and spin and uses only the outer success bit.
    """
    _, rank, _ = _df_address_widths(spin_orbitals, eigenvalues)
    body = source_double_factorization_one_body_cost(
        spin_orbitals, eigenvalues, alias_bits, basis_rotation_bits, inner_rotation_bits
    )
    width = body.qubits + _auxiliary_width(auxiliaries) + 1
    prep = source_double_factorization_outer_prepare_cost(auxiliaries, alias_bits)
    unprep = source_double_factorization_outer_prepare_cost(
        auxiliaries, alias_bits, adjoint=True
    )
    indexed = (auxiliaries, spin_orbitals, eigenvalues, inner_rotation_bits)
    load = source_double_factorization_indexed_data_cost(*indexed)
    unload = source_double_factorization_indexed_data_cost(*indexed, adjoint=True)
    reflected = rank + 1
    identity = ComponentCost("PrepareIdentity", reflected)
    reflection = source_prepare_reflection_cost(
        reflected, 0, identity, identity, controlled=True
    )
    return ComponentCost(
        "DoubleFactorizationBlockEncoding",
        width,
        calls=tuple(
            CostCall(child, bystander_qubits=width - child.qubits)
            for child in (prep, load, body, reflection, body, unload, unprep)
        ),
    )
