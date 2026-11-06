# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 grid potential call graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/potential.py

"""Locked grid potential model with the original data and width conventions."""

from itertools import combinations

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.arithmetic_resource_model import source_sum_of_squares_cost
from ...compiler.registers.qrom_resource_model import source_qrom_cost
from .grid_inverse_sqrt import (
    grid_inverse_sqrt_qrom_data,
    source_grid_inverse_sqrt_newton_cost,
    source_grid_inverse_sqrt_polynomial_cost,
)
from .grid_kinetic import _coordinate_width
from .grid_qvr import source_grid_qvr_cost


def source_grid_pair_potential_cost(
    coordinate: int,
    polynomial: int = 15,
    target: int = 24,
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Model the source pair graph, retaining its shorter coefficient table."""
    if type(coordinate) is not int or coordinate < 1:
        raise ValueError("Grid pair potential requires a positive coordinate width")
    distance = 2 * (coordinate + 1) + 2
    newton = source_grid_inverse_sqrt_newton_cost(
        distance, polynomial, target, legacy_shims=legacy_shims
    )
    poly = source_grid_inverse_sqrt_polynomial_cost(
        distance, polynomial, polynomial, legacy_shims=legacy_shims
    )
    data = grid_inverse_sqrt_qrom_data(distance - 2, polynomial)
    lookup = source_qrom_cost(
        0,
        (distance,),
        (len(data[0]),),
        ((1, polynomial),) * 4,
        data,
        legacy_shims=legacy_shims,
    )
    system = 6 * coordinate
    difference = coordinate + 1
    addition = ComponentCost(
        "OutOfPlaceAdder",
        2 * coordinate + difference,
        GateCost(and_bloq=coordinate, clifford=5 * coordinate),
    )
    erase = ComponentCost(
        "OutOfPlaceAdder",
        addition.qubits,
        GateCost(
            clifford=(8 if legacy_shims else 6) * coordinate, measurement=coordinate
        ),
    )
    # The source greedy schedule returns the system wires before the
    # inverse-distance stages and releases coefficients after the cubic.
    calls = (
        CostCall(addition, 3, system - 2 * coordinate + 2 * difference),
        CostCall(source_sum_of_squares_cost(difference, 3), bystander_qubits=system),
        CostCall(erase, 3, system - 2 * coordinate + 2 * difference + distance),
        CostCall(ComponentCost("Cast", distance)),
        CostCall(ComponentCost("Allocate", polynomial), 4),
        CostCall(lookup),
        CostCall(ComponentCost("Cast", distance)),
        CostCall(ComponentCost("Allocate", polynomial)),
        CostCall(poly),
        CostCall(ComponentCost("Free", polynomial), 4),
        CostCall(ComponentCost("Allocate", target)),
        CostCall(newton),
        CostCall(ComponentCost("Free", distance)),
        CostCall(ComponentCost("Free", polynomial)),
        CostCall(source_grid_qvr_cost(target)),
        CostCall(ComponentCost("Free", target)),
    )
    return ComponentCost("PairPotential", system, calls=calls)


def source_grid_potential_cost(
    electrons: int,
    grid_points: int,
    polynomial: int = 15,
    target: int = 24,
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Model one source call for every upper-triangular electron pair."""
    if type(electrons) is not int or electrons < 1:
        raise ValueError("Grid potential requires a positive electron count")
    coordinate = _coordinate_width(grid_points)
    system = 3 * electrons * coordinate
    if electrons == 1:
        return ComponentCost("PotentialEnergy", system)
    pair = source_grid_pair_potential_cost(
        coordinate, polynomial, target, legacy_shims=legacy_shims
    )
    edge = CostCall(pair, bystander_qubits=system - pair.qubits)
    return ComponentCost(
        "PotentialEnergy",
        system,
        calls=tuple(edge for _ in combinations(range(electrons), 2)),
    )
