# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: locked Qualtran v0.7.0 chemistry/sf and chemistry/df cost graphs.

"""Extend owned concrete child budgets with literal symbolic source terms."""

from __future__ import annotations

from typing import cast

from ...compiler._component_trace.qec_gate_cost import QECGateCost
from ...compiler._component_trace.resource_model import estimate_component_resources
from ...compiler.registers.comparison.comparison_resource_model import (
    source_less_equal_resource_formula,
)
from ...compiler.registers.comparison.less_equal import (
    source_equal_width_less_equal_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...compiler.registers.sorting.contiguous_index import source_contiguous_index_cost
from ...core.value_math import SymbolicInt, smax
from .double_factorization_encoding_model import (
    _double_factorization_one_body_declaration,
    source_double_factorization_one_body_cost,
)
from .double_factorization_prepare_model import _double_factorization_inner_declaration
from .single_factorization_encoding_model import (
    source_single_factorization_one_body_cost,
    source_single_factorization_one_body_signature,
)
from .single_factorization_resource_model import (
    source_single_factorization_inner_prepare_signature,
)


def _formula(gates: QECGateCost, peak: SymbolicInt) -> SymbolicResourceFormula:
    return SymbolicResourceFormula(
        gates.and_bloq,
        gates.clifford,
        gates.measurement,
        peak,
        t=gates.t,
        toffoli=gates.toffoli,
        cswap=gates.cswap,
        rotation=gates.rotation,
    )


def source_single_factorization_one_body_formula(
    auxiliaries: int,
    spin_orbitals: int,
    alias_bits: SymbolicInt = 8,
    rotation_bits: SymbolicInt = 8,
    *,
    forward_blocks: tuple[int, int] = (1, 1),
    inverse_blocks: tuple[int, int] = (1, 1),
) -> SymbolicResourceFormula:
    """Keep symbolic comparison dispatch and both fixed QROAM block choices.

    Unchanged concrete children retain their authoritative models. Inner PREPARE
    changes only its explicit Toffoli term, fixed-bank output cost, comparison
    declaration and QROAM/inequality width bounds. Source is_adjoint does not
    affect the one-body call graph.
    """
    baseline_alias = alias_bits if type(alias_bits) is int else 8
    baseline_rotation = rotation_bits if type(rotation_bits) is int else 8
    model = source_single_factorization_one_body_cost(
        auxiliaries,
        spin_orbitals,
        baseline_alias,
        baseline_rotation,
        forward_blocks=forward_blocks,
        inverse_blocks=inverse_blocks,
    )
    report = estimate_component_resources(model)
    if type(alias_bits) is int and type(rotation_bits) is int:
        return _formula(QECGateCost().plus(report.gates, 1), report.peak_logical_qubits)
    comparison = source_less_equal_resource_formula(alias_bits, alias_bits)
    previous = source_equal_width_less_equal_cost(baseline_alias)
    gates = (
        QECGateCost().plus(report.gates, 1).plus(previous.gates, -2).plus(comparison, 2)
    )
    gates += QECGateCost(
        toffoli=(alias_bits - baseline_alias)
        * (forward_blocks[0] * forward_blocks[1] - 1)
        + 4 * (rotation_bits - baseline_rotation)
    )
    rank = (spin_orbitals // 2 - 1).bit_length()
    inner_ports = source_single_factorization_inner_prepare_signature(
        auxiliaries, spin_orbitals
    ).n_qubits()
    contiguous = estimate_component_resources(
        source_contiguous_index_cost(rank, 2 * rank)
    )
    qroam_ports = (auxiliaries).bit_length() + 2 * rank + alias_bits + 2
    inner_peak = smax(
        inner_ports,
        3,
        contiguous.peak_logical_qubits,
        4 * rank + 1,
        qroam_ports,
        comparison.logical_qubits,
    )
    ports = source_single_factorization_one_body_signature(
        auxiliaries, spin_orbitals
    ).n_qubits()
    other_peaks = tuple(
        estimate_component_resources(call.component).peak_logical_qubits
        + call.bystander_qubits
        for call in model.calls[1:-1]
    )
    peak = smax(ports, inner_peak + ports - inner_ports, *other_peaks)
    return _formula(gates, cast(SymbolicInt, peak))


def source_double_factorization_one_body_formula(
    spin_orbitals: int,
    eigenvalues: int,
    alias_bits: int = 8,
    basis_rotation_bits: int = 24,
    inner_rotation_bits: SymbolicInt = 8,
) -> SymbolicResourceFormula:
    """Retain symbolic inner rotation storage and literal declared phase cost.

    Alias/basis precision still uses the original numerical QROAM optimization;
    an unresolved argument there has no successful source category/peak query.
    The exact root adjoint has the same category sum and through-word live bound.
    """
    baseline = inner_rotation_bits if type(inner_rotation_bits) is int else 8
    model = source_double_factorization_one_body_cost(
        spin_orbitals, eigenvalues, alias_bits, basis_rotation_bits, baseline
    )
    report = estimate_component_resources(model)
    gates = QECGateCost().plus(report.gates, 1) + QECGateCost(
        toffoli=4 * (inner_rotation_bits - baseline)
    )
    if type(inner_rotation_bits) is int:
        return _formula(gates, report.peak_logical_qubits)
    ports = _double_factorization_one_body_declaration(
        spin_orbitals, eigenvalues, basis_rotation_bits, inner_rotation_bits
    ).n_qubits()
    inner_ports = _double_factorization_inner_declaration(
        spin_orbitals, eigenvalues, inner_rotation_bits
    ).n_qubits()
    inner = model.calls[0].component
    inner_peak = smax(
        inner_ports,
        *(
            estimate_component_resources(call.component).peak_logical_qubits
            for call in inner.calls
        ),
    )
    other_peaks = tuple(
        estimate_component_resources(call.component).peak_logical_qubits
        + call.bystander_qubits
        + ports
        - model.qubits
        for index, call in enumerate(model.calls)
        if index not in (0, 7)
    )
    peak = smax(ports, inner_peak + ports - inner_ports, *other_peaks)
    return _formula(gates, cast(SymbolicInt, peak))


def source_single_factorization_block_formula(
    auxiliaries: int,
    spin_orbitals: int,
    alias_bits: int = 8,
    inner_rotation_bits: SymbolicInt = 8,
    *,
    forward_blocks: tuple[int, int] = (1, 1),
    inverse_blocks: tuple[int, int] = (1, 1),
) -> SymbolicResourceFormula:
    """Compose the two literal one-body calls and fixed outer/reflection owners."""
    from .single_factorization_encoding_model import (
        source_single_factorization_block_encoding_cost,
    )

    baseline = inner_rotation_bits if type(inner_rotation_bits) is int else 8
    root = source_single_factorization_block_encoding_cost(
        auxiliaries,
        spin_orbitals,
        alias_bits,
        baseline,
        forward_blocks=forward_blocks,
        inverse_blocks=inverse_blocks,
    )
    report = estimate_component_resources(root)
    body = source_single_factorization_one_body_formula(
        auxiliaries,
        spin_orbitals,
        alias_bits,
        inner_rotation_bits,
        forward_blocks=forward_blocks,
        inverse_blocks=inverse_blocks,
    )
    previous = estimate_component_resources(root.calls[1].component)
    gates = QECGateCost().plus(report.gates, 1).plus(previous.gates, -2).plus(body, 2)
    peak = smax(
        root.qubits,
        *(
            (
                body.logical_qubits + call.bystander_qubits
                if index in (1, 3)
                else estimate_component_resources(call.component).peak_logical_qubits
                + call.bystander_qubits
            )
            for index, call in enumerate(root.calls)
        ),
    )
    return _formula(gates, cast(SymbolicInt, peak))
