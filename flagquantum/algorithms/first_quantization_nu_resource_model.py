# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareNuState and its subcomponents.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py

"""Resource models for preparing the first-quantized momentum transfer."""

from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..compiler.registers.arithmetic_resource_model import (
    source_product_cost,
    source_sum_of_squares_cost,
)
from ..compiler.registers.ordering import source_greater_cost
from ..compiler.registers.uniform_preparation import source_uniform_preparation_cost


def _inequality_cost(p: int, m: int, adjoint: bool) -> ComponentCost:
    ports = 4 * p + m + 7
    if adjoint:
        return ComponentCost("TestNuInequality", ports)
    result_width = 2 * p + 2
    return ComponentCost(
        "TestNuInequality",
        ports,
        GateCost(toffoli=3),
        (
            CostCall(source_sum_of_squares_cost(p, 3)),
            CostCall(source_product_cost(result_width, m)),
            CostCall(source_greater_cost(m, result_width)),
        ),
    )


def source_prepare_nu_cost(
    num_bits_p: int,
    m_param: int,
    *,
    adjoint: bool = False,
    decomposed: bool = False,
) -> ComponentCost:
    """Model the declared gates or the source composite, with the same peak.

    The big-endian mu and three nu registers, the m word and the success
    flag pass through. Three flags produced internally are consumed by the
    inequality in the forward direction. Only uniform m has gate emission.
    """
    if type(num_bits_p) is not int or num_bits_p < 1:
        raise ValueError("Momentum width must be a positive integer")
    if type(m_param) is not int or m_param < 2:
        raise ValueError("Source momentum transfer requires m_param >= 2")
    p, m = num_bits_p, (m_param - 1).bit_length()
    nu_width = 3 * (p + 1)
    ports = p + nu_width + m + 1
    uniform = source_uniform_preparation_cost(m_param)
    m_call = CostCall(uniform, bystander_qubits=ports - m)
    # In Qualtran's width schedule the independent m preparation can run
    # before the three flags exist, or after they have been consumed.
    width = ports if decomposed else max(ports, uniform.qubits + ports - m)
    calls: tuple[CostCall, ...] = (
        CostCall(
            ComponentCost(
                "PrepareMuUnaryEncodedOneHot", max(p + 1, 3), GateCost(toffoli=p - 1)
            ),
            bystander_qubits=ports - p,
        ),
        CostCall(
            ComponentCost(
                "PrepareNuSuperPositionState",
                p + nu_width,
                GateCost(toffoli=3 * (p - 1)),
            ),
            bystander_qubits=m + 2,
        ),
        CostCall(
            ComponentCost(
                "FlagZeroAsFailure",
                nu_width + 1,
                GateCost(toffoli=0 if adjoint else 3 * p + 2),
            ),
            bystander_qubits=p + m + 2,
        ),
        CostCall(
            ComponentCost(
                "TestNuLessThanMu",
                p + nu_width + 1,
                GateCost(toffoli=0 if adjoint else 3 * p),
            ),
            bystander_qubits=m + 3,
        ),
        CostCall(_inequality_cost(p, m, adjoint)),
    )
    if decomposed:
        calls = (m_call, *calls)
    return ComponentCost(
        "PrepareNuState†" if adjoint else "PrepareNuState",
        width,
        calls=calls[::-1] if adjoint else calls,
    )
