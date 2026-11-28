# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: locked Qualtran v0.7.0 first_quantization/prepare_uv.py and prepare_zeta.py.

"""Compose literal UV declaration estimates without assigning physical bodies."""

from __future__ import annotations

from typing import cast

from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ..core.value_math import SymbolicInt, smax
from .first_quantization_nu_resource_model import (
    source_nu_gate_cost,
    source_nu_resource_formula,
)


def source_uv_gate_cost(
    num_bits_p: SymbolicInt,
    m_param: int,
    lambda_zeta: SymbolicInt,
    *,
    num_bits_n: SymbolicInt | None = None,
    adjoint: bool = False,
) -> QECGateCost:
    """Retain nu plus zeta, including the original numerical inverse rounding."""
    if adjoint:
        import numpy as np

        charge: SymbolicInt = int(np.ceil(lambda_zeta**0.5))
    else:
        charge = lambda_zeta
    return source_nu_gate_cost(
        num_bits_p, m_param, num_bits_n=num_bits_n, adjoint=adjoint
    ) + QECGateCost(toffoli=charge)


def source_uv_resource_formula(
    num_bits_p: SymbolicInt,
    num_atoms: int,
    m_param: int,
    lambda_zeta: SymbolicInt,
    *,
    num_bits_n: SymbolicInt | None = None,
    adjoint: bool = False,
) -> SymbolicResourceFormula:
    """Keep disjoint index bystanders live across both original child calls.

    eta and nuclear position precision have no source effect here. Opaque nu
    and zeta leaves do not acquire primitive recipes from these estimates.
    Invalid projectile inverse wiring still prevents a source peak query.
    """
    nu = source_nu_resource_formula(
        num_bits_p, m_param, num_bits_n=num_bits_n, adjoint=adjoint
    )
    gates = source_uv_gate_cost(
        num_bits_p, m_param, lambda_zeta, num_bits_n=num_bits_n, adjoint=adjoint
    )
    index = (num_atoms - 1).bit_length()
    width = num_bits_p if num_bits_n is None else num_bits_n
    momentum = 4 * width + (m_param - 1).bit_length() + 4
    peak = cast(SymbolicInt, smax(nu.logical_qubits + index, smax(index, 3) + momentum))
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
