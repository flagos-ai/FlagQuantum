# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's controlled HamiltonianSimulationByGQSP.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/hamiltonian_simulation/hamiltonian_simulation_by_gqsp.py

"""Declared resource queries for plain and externally controlled Hubbard GQSP."""

from dataclasses import fields

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ...compiler.registers.preparation.prepare_reflection import (
    source_prepare_reflection_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.value_math import SymbolicReal
from ...simulation.numerics.jacobi_anger_degree import degree_jacobi_anger_approximation
from .hubbard_prepare import PrepareHubbard
from .hubbard_select import SelectHubbard


def _extend_control(child: ComponentCost, name: str) -> ComponentCost:
    """Add the source two-bit conjunction around a positive-controlled child."""
    return ComponentCost(
        name,
        child.qubits + 1,
        GateCost(and_bloq=1, clifford=1, measurement=1),
        (CostCall(child, bystander_qubits=2),),
    )


def _hubbard_stages(
    prepare: PrepareHubbard,
) -> tuple[ComponentCost, ComponentCost, ComponentCost]:
    prep = prepare.source_resource_model()
    select = SelectHubbard(prepare.dimension, controlled=True).source_resource_model()
    selection = sum(width for _, width in prepare.selection_layout)
    reflection = source_prepare_reflection_cost(
        selection, prepare.layout[-1][1], prep, prep, controlled=True
    )
    return prep, select, reflection


def _positive_walk(
    select: ComponentCost, reflection: ComponentCost, system: int
) -> ComponentCost:
    return ComponentCost(
        "PositiveControlledHubbardWalk",
        select.qubits,
        calls=(CostCall(select), CostCall(reflection, bystander_qubits=system)),
    )


def source_hubbard_gqsp_formula(
    prepare: PrepareHubbard,
    alpha: SymbolicReal,
    time: SymbolicReal,
    precision: SymbolicReal,
) -> SymbolicResourceFormula:
    """Aggregate the original plain walk, including its symbolic Jacobi–Anger degree."""
    degree = degree_jacobi_anger_approximation(time * alpha, precision=precision / 2)
    prep, select, reflection = _hubbard_stages(prepare)
    system = 2 * prepare.dimension**2
    positive = estimate_component_resources(_positive_walk(select, reflection, system))
    preparation = estimate_component_resources(prep)
    # The forward zero control adds two X gates. The inverse has the same
    # positive-control resource totals as its forward child.
    per_degree = positive.gates.plus(positive.gates, 1).plus(
        GateCost(clifford=2, rotation=6), 1
    )
    constant = preparation.gates.plus(preparation.gates, 1).plus(
        GateCost(rotation=3), 1
    )
    return SymbolicResourceFormula(
        logical_qubits=max(
            positive.peak_logical_qubits + prepare.layout[-1][1],
            preparation.peak_logical_qubits + system + 1,
        ),
        **{
            category.name: getattr(per_degree, category.name) * degree
            + getattr(constant, category.name)
            for category in fields(GateCost)
        },
    )


def source_controlled_hubbard_gqsp_cost(
    prepare: PrepareHubbard, time: float, precision: float
) -> ComponentCost:
    """Keep outer PREPAREs unconditional and model the two controlled walk paths."""
    degree = degree_jacobi_anger_approximation(
        time * prepare.alpha, precision=precision / 2
    )
    assert isinstance(degree, int)
    prep, select, reflection = _hubbard_stages(prepare)
    system = 2 * prepare.dimension**2
    selection = sum(width for _, width in prepare.selection_layout)
    junk = prepare.layout[-1][1]
    positive = _positive_walk(select, reflection, system)
    forward = _extend_control(positive, "ControlledZeroControlledHubbardWalk")
    # General control of the inverse walk reaches its SELECT and reflection
    # separately; the source does not wrap the whole inverse in one predicate.
    inverse = ComponentCost(
        "ControlledPositiveControlledHubbardWalk†",
        selection + system + 2,
        calls=(
            CostCall(
                _extend_control(reflection, "DoubleControlledReflection"),
                bystander_qubits=system,
            ),
            CostCall(_extend_control(select, "DoubleControlledSelect†")),
        ),
    )
    crz = ComponentCost("CRz", 2, GateCost(clifford=2, rotation=2))
    rotation = ComponentCost(
        "ControlledSU2Rotation",
        2,
        calls=(
            CostCall(
                ComponentCost("ControlledScalarZPow", 1, GateCost(rotation=1)),
                bystander_qubits=1,
            ),
            CostCall(crz, 2),
            CostCall(ComponentCost("ControlledRx", 2, GateCost(rotation=1))),
        ),
    )
    return ComponentCost(
        "ControlledHamiltonianSimulationByGQSP",
        selection + system + 2,
        calls=(
            CostCall(prep, bystander_qubits=system + 2),
            CostCall(prep, bystander_qubits=system + 2),
            CostCall(forward, degree, bystander_qubits=junk),
            CostCall(inverse, degree, bystander_qubits=junk),
            CostCall(
                rotation, 2 * degree + 1, bystander_qubits=selection + system + junk
            ),
        ),
    )
