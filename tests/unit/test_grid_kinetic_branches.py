"""Certify complete native kinetic diagonal channels over every small basis word."""

from cmath import exp
from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.grid_kinetic import GridKineticEnergy
from flagquantum.compiler.registers.arithmetic.integer_square import (
    sum_of_squares_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_monomial_map

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("electrons,coordinate", ((1, 1), (2, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_full_native_basis_phase_workzero_and_all_combined_histories(
    electrons: int, coordinate: int, adjoint: bool
) -> None:
    system_width = 3 * electrons * coordinate
    result_width, work_width = sum_of_squares_requirements(coordinate, 3)
    result = tuple(range(system_width, system_width + result_width))
    workspace = tuple(
        range(system_width + result_width, system_width + result_width + work_width)
    )
    spectator = system_width + result_width + work_width
    system = tuple(
        tuple(
            tuple(range((3 * e + axis) * coordinate, (3 * e + axis + 1) * coordinate))
            for axis in range(3)
        )
        for e in range(electrons)
    )
    coefficient = 0.317
    operation = GridKineticEnergy.make_on(
        grid_points=1 << (coordinate - 1),
        system=system,
        result=result,
        workspace=workspace,
        radians_per_unit=coefficient,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(spectator + 1).append(operation)
    gates = circuit.to_ir().instructions
    inputs = []
    phases = []
    mask = (1 << coordinate) - 1
    for data, bystander in product(range(1 << system_width), (0, 1)):
        inputs.append(
            tuple(data >> bit & 1 for bit in reversed(range(system_width)))
            + (0,) * (result_width + work_width)
            + (bystander,)
        )
        squared = sum(
            ((data >> (coordinate * axis)) & mask) ** 2 for axis in range(3 * electrons)
        )
        phases.append(exp((-1 if adjoint else 1) * 1j * coefficient * squared))
    observed, actual_phases, measurements = certified_and_monomial_map(
        gates, tuple(inputs)
    )
    assert observed == tuple(inputs)
    torch.testing.assert_close(
        actual_phases, torch.tensor(phases, dtype=torch.complex128), atol=3e-12, rtol=0
    )
    assert measurements == sum(g.name == "measure" for g in gates)
    assert measurements > 0
    # Literal And blocks and their actual erased conjunctions prove this phased
    # diagonal map for every combined history with scalar 2**(-measurements/2).
    # Full basis columns and the spectator establish every complex input state.
