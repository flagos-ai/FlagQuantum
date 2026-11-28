# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: locked Qualtran v0.7.0 ordinary/projectile select_and_prepare.py.

"""Compose source PREPARE budgets without turning declarations into circuits."""

from __future__ import annotations

from typing import cast

from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ..core.value_math import SymbolicInt, smax
from .first_quantization import _first_quantization_declaration
from .first_quantization_potential_formula import (
    source_uv_gate_cost,
    source_uv_resource_formula,
)
from .first_quantization_projectile import _projectile_prepare_declaration


def source_first_quantization_prepare_gate_cost(
    num_bits_p: SymbolicInt,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 256,
    num_bits_n: SymbolicInt | None = None,
    num_bits_t: SymbolicInt = 16,
    num_bits_rot_aa: SymbolicInt = 8,
    adjoint: bool = False,
) -> QECGateCost:
    """Query literal four-child categories, including omitted uniform budgets."""
    width = num_bits_p if num_bits_n is None else num_bits_n
    particle = 7 * (eta - 1).bit_length() + 4 * num_bits_rot_aa - 18
    if num_bits_n is None:
        term = (
            num_bits_t
            + 4 * (eta + 2 * lambda_zeta - 1).bit_length()
            + 2 * num_bits_rot_aa
            - 12
        )
        kinetic = 13 + 2 * (width - 2)
    else:
        term = 0 if adjoint else 6 * num_bits_t + 2
        power = width - 2 if adjoint else 2 * width - num_bits_p - 2
        kinetic = (15 if adjoint else 31) + 2 * power
    potential = source_uv_gate_cost(
        num_bits_p, m_param, lambda_zeta, num_bits_n=num_bits_n, adjoint=adjoint
    )
    return potential + QECGateCost(toffoli=term + particle + kinetic)


def source_first_quantization_prepare_formula(
    num_bits_p: SymbolicInt,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 256,
    num_bits_n: SymbolicInt | None = None,
    num_bits_t: SymbolicInt = 16,
    num_bits_rot_aa: SymbolicInt = 8,
    adjoint: bool = False,
) -> SymbolicResourceFormula:
    """Retain complete bystanders and original failed inverse peak queries."""
    gates = source_first_quantization_prepare_gate_cost(
        num_bits_p,
        eta,
        num_atoms,
        lambda_zeta,
        m_param=m_param,
        num_bits_n=num_bits_n,
        num_bits_t=num_bits_t,
        num_bits_rot_aa=num_bits_rot_aa,
        adjoint=adjoint,
    )
    if eta < 2 or (num_bits_n is None and num_bits_p == 0):
        raise ValueError("Original PREPARE has no valid complete call graph")
    width = num_bits_p if num_bits_n is None else num_bits_n
    signature = (
        _first_quantization_declaration(width, eta, num_atoms, m_param, prepare=True)
        if num_bits_n is None
        else _projectile_prepare_declaration(width, eta, num_atoms, m_param)
    )
    ports = signature.n_qubits()
    particles = 2 * (eta - 1).bit_length()
    potential_ports = (
        4 * width + (m_param - 1).bit_length() + 4 + (num_atoms - 1).bit_length()
    )
    potential = source_uv_resource_formula(
        num_bits_p,
        num_atoms,
        m_param,
        lambda_zeta,
        num_bits_n=num_bits_n,
        adjoint=adjoint,
    )
    if num_bits_n is None:
        kinetic_ports = 2 + 2 * width
        kinetic_peak = smax(kinetic_ports + 4, width + smax(width, 3) + 2)
        term_ports = 2
    else:
        kinetic_ports = 4 + 2 * width
        kinetic_peak = smax(kinetic_ports, 3)
        term_ports = 9
    peak = smax(
        ports,
        ports - term_ports + smax(term_ports, 3),
        ports - particles + smax(particles, 3),
        ports - kinetic_ports + kinetic_peak,
        ports - potential_ports + potential.logical_qubits,
    )
    return SymbolicResourceFormula(
        gates.and_bloq,
        gates.clifford,
        gates.measurement,
        cast(SymbolicInt, peak),
        t=gates.t,
        toffoli=gates.toffoli,
        cswap=gates.cswap,
        rotation=gates.rotation,
    )
