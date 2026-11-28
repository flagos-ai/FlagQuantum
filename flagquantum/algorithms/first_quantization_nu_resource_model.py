# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareNuState, its projectile variant and subcomponents.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py

# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_nu.py

"""Resource models for preparing the first-quantized momentum transfer."""

from __future__ import annotations

from typing import cast

from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ..compiler.registers.arithmetic.arithmetic_resource_model import (
    source_product_cost,
    source_product_gate_cost,
    source_sum_of_squares_cost,
    source_sum_of_squares_gate_cost,
)
from ..compiler.registers.comparison.comparison_resource_model import (
    source_less_equal_gate_cost,
    source_less_equal_resource_formula,
)
from ..compiler.registers.comparison.ordering import source_greater_cost
from ..compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from ..compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ..core.value_math import SymbolicInt, smax
from .first_quantization_nu import (
    prepare_nu_leaf_signatures,
    prepare_nu_signature,
    projectile_prepare_mu_signature,
)


def _inequality_cost(p: int, m: int, ports: int, adjoint: bool) -> ComponentCost:
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
    p = num_bits_p
    mu = ComponentCost(
        "PrepareMuUnaryEncodedOneHot", max(p + 1, 3), GateCost(toffoli=p - 1)
    )
    return _prepare_nu_cost(
        p, m_param, mu, "PrepareNuState", adjoint=adjoint, decomposed=decomposed
    )


def _prepare_nu_cost(
    p: int,
    m_param: int,
    mu: ComponentCost,
    name: str,
    *,
    adjoint: bool,
    decomposed: bool,
) -> ComponentCost:
    if type(m_param) is not int or m_param < 2:
        raise ValueError("Source momentum transfer requires m_param >= 2")
    m = (m_param - 1).bit_length()
    ports = int(prepare_nu_signature(p, m_param).n_qubits())
    leaf_widths = {
        name: int(signature.n_qubits())
        for name, signature in prepare_nu_leaf_signatures(
            p, m_param, adjoint=adjoint
        ).items()
    }
    uniform = source_uniform_preparation_cost(m_param)
    m_call = CostCall(uniform, bystander_qubits=ports - m)
    # In Qualtran's width schedule the independent m preparation can run
    # before the three flags exist, or after they have been consumed.
    width = ports if decomposed else max(ports, uniform.qubits + ports - m)
    calls: tuple[CostCall, ...] = (
        CostCall(mu, bystander_qubits=ports - p),
        CostCall(
            ComponentCost(
                "PrepareNuSuperPositionState",
                leaf_widths["PrepareNuSuperPositionState"],
                _nu_superposition_gates(p),
            ),
            bystander_qubits=m + 2,
        ),
        CostCall(
            ComponentCost(
                "FlagZeroAsFailure",
                leaf_widths["FlagZeroAsFailure"],
                GateCost(toffoli=0 if adjoint else 3 * p + 2),
            ),
            bystander_qubits=p + m + 2,
        ),
        CostCall(
            ComponentCost(
                "TestNuLessThanMu",
                leaf_widths["TestNuLessThanMu"],
                GateCost(toffoli=0 if adjoint else 3 * p),
            ),
            bystander_qubits=m + 3,
        ),
        CostCall(_inequality_cost(p, m, leaf_widths["TestNuInequality"], adjoint)),
    )
    if decomposed:
        calls = (m_call, *calls)
    return ComponentCost(
        f"{name}†" if adjoint else name,
        width,
        calls=calls[::-1] if adjoint else calls,
    )


def _nu_superposition_gates(width: int) -> GateCost:
    return GateCost(toffoli=3 * (width - 1))


def source_projectile_prepare_mu_cost(
    num_bits_p: int, num_bits_n: int, *, adjoint: bool = False
) -> ComponentCost:
    """Estimate the projectile unary word with its source right-only flag."""
    signature = projectile_prepare_mu_signature(num_bits_p, num_bits_n)
    toffoli = source_projectile_prepare_mu_formula(
        num_bits_p, num_bits_n, adjoint=adjoint
    ).toffoli
    assert isinstance(toffoli, int)
    return ComponentCost(
        "PrepareMuUnaryEncodedOneHotWithProj",
        max(int(signature.n_qubits()), 3),
        GateCost(toffoli=toffoli),
    )


def source_projectile_prepare_mu_formula(
    num_bits_p: SymbolicInt, num_bits_n: SymbolicInt, *, adjoint: bool = False
) -> SymbolicResourceFormula:
    """Retain the opaque mu declaration, including its unchanged RIGHT flag."""
    if num_bits_n - num_bits_p < 0:
        raise ValueError("Projectile width is less than electron width")
    return SymbolicResourceFormula(
        0,
        0,
        0,
        cast(SymbolicInt, smax(num_bits_n + 1, 3)),
        toffoli=num_bits_n if adjoint else 2 * num_bits_n - num_bits_p - 1,
    )


def source_projectile_prepare_nu_cost(
    num_bits_p: int, num_bits_n: int, m_param: int, *, decomposed: bool = False
) -> ComponentCost:
    """Report the projectile forward declaration or forward composite cost."""
    mu = source_projectile_prepare_mu_cost(num_bits_p, num_bits_n)
    return _prepare_nu_cost(
        num_bits_n,
        m_param,
        mu,
        "PrepareNuStateWithProj",
        adjoint=False,
        decomposed=decomposed,
    )


def source_projectile_prepare_nu_adjoint_gate_cost(
    num_bits_p: int, num_bits_n: int
) -> GateCost:
    """Count declared adjoint gates. The source adjoint has no valid peak width."""
    mu = source_projectile_prepare_mu_cost(num_bits_p, num_bits_n, adjoint=True)
    # The other three children have empty adjoint gate declarations.
    return mu.gates.plus(_nu_superposition_gates(num_bits_n), 1)


def projectile_prepare_nu_adjoint_model(
    num_bits_p: int, num_bits_n: int, m_param: int, *, decomposed: bool = False
) -> ComponentCost:
    """Estimate the native adjoint with the μ flag consumed on the left.

    This uses the declared adjoint gate counts and native live widths.
    Its peak is an estimate for the corrected binding, since the frozen
    source's right-only flag prevents its own adjoint width query.
    """
    mu = source_projectile_prepare_mu_cost(num_bits_p, num_bits_n, adjoint=True)
    return _prepare_nu_cost(
        num_bits_n,
        m_param,
        mu,
        "PrepareNuStateWithProj",
        adjoint=True,
        decomposed=decomposed,
    )


def source_nu_gate_cost(
    num_bits_p: SymbolicInt,
    m_param: int,
    *,
    num_bits_n: SymbolicInt | None = None,
    adjoint: bool = False,
) -> QECGateCost:
    """Keep literal nu categories, including symbolic widths and omitted uniform m.

    num_bits_n selects the projectile declaration. Gate counts alone do not
    certify physical bodies or the projectile inverse's invalid flag wiring.
    """
    if type(m_param) is not int:
        raise ValueError("Source momentum declarations require integer m_param")
    width = num_bits_p if num_bits_n is None else num_bits_n
    if num_bits_n is not None and width - num_bits_p < 0:
        raise ValueError("Projectile width is less than electron width")
    mu = (
        width - 1
        if num_bits_n is None
        else source_projectile_prepare_mu_formula(
            num_bits_p, width, adjoint=adjoint
        ).toffoli
    )
    gates = QECGateCost(toffoli=mu + 3 * (width - 1))
    if adjoint:
        return gates
    sample_width = (m_param - 1).bit_length()
    product_width = 2 * width + 2
    gates += QECGateCost(toffoli=6 * width + 5)
    gates += source_sum_of_squares_gate_cost(width, 3)
    gates += source_product_gate_cost(product_width, sample_width)
    comparison = source_less_equal_gate_cost(sample_width, product_width)
    return gates.plus(comparison, 1) + QECGateCost(clifford=1)


def source_nu_resource_formula(
    num_bits_p: SymbolicInt,
    m_param: int,
    *,
    num_bits_n: SymbolicInt | None = None,
    adjoint: bool = False,
) -> SymbolicResourceFormula:
    """Query the literal declared categories and complete source live-width bound.

    The original projectile inverse has no valid width schedule. Its categories
    remain available separately through source_nu_gate_cost. Uniform m contributes
    to the source live peak even though the analytical gate graph omits it.
    """
    if num_bits_n is not None and adjoint:
        raise ValueError("Original projectile nu adjoint has no valid peak width")
    if m_param <= 1:
        raise ValueError("Original nu sample body has no valid peak width")
    gates = source_nu_gate_cost(
        num_bits_p, m_param, num_bits_n=num_bits_n, adjoint=adjoint
    )
    width = num_bits_p if num_bits_n is None else num_bits_n
    sample = (m_param - 1).bit_length()
    ports = 4 * width + sample + 4
    inequality: SymbolicInt = 4 * width + sample + 7
    if not adjoint:
        product = 2 * width + 2
        greater = (
            estimate_component_resources(
                source_greater_cost(sample, product)
            ).peak_logical_qubits
            if type(width) is int
            else source_less_equal_resource_formula(sample, product).logical_qubits
        )
        inequality = cast(
            SymbolicInt,
            smax(
                inequality,
                5 * width + 2,
                product + sample + 2 * max(product, sample),
                greater,
                3,
            ),
        )
    uniform = estimate_component_resources(source_uniform_preparation_cost(m_param))
    peak = cast(
        SymbolicInt,
        smax(
            ports,
            smax(width + 1, 3) + ports - width,
            smax(4 * width + 3, 3) + sample + 2,
            smax(3 * width + 4, 3) + width + sample + 2,
            smax(4 * width + 4, 3) + sample + 3,
            inequality,
            uniform.peak_logical_qubits + ports - sample,
        ),
    )
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
