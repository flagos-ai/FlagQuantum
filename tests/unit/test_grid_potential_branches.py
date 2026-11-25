"""Certify complete native pair/total potential channels and erasure histories."""

from cmath import exp
from collections.abc import Callable
from fractions import Fraction
from itertools import combinations, product
from typing import cast

import pytest
import torch

from flagquantum.algorithms.primitives.grid_potential import (
    GridPairPotential,
    GridPotentialEnergy,
    grid_potential_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_monomial_map
from tests.unit.test_grid_cubic_inverse_sqrt import cubic_word
from tests.unit.test_grid_potential import _estimate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "electrons,cubic,precision,iterations",
    ((2, False, 4, 0), (2, False, 4, 1), (3, False, 4, 1), (2, True, 5, 1)),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_native_diagonal_columns_and_every_combined_history(
    electrons: int, cubic: bool, precision: int, iterations: int, adjoint: bool
) -> None:
    logical = 3 * electrons
    work, _ = grid_potential_requirements(
        1, precision=precision, iterations=iterations, cubic=cubic
    )
    system = tuple(
        tuple((3 * i + axis,) for axis in range(3)) for i in range(electrons)
    )
    theta = 0.721
    workspace = tuple(range(logical, logical + work))
    # Existing untyped scenario oracle remains the independent rational spec.
    estimate = cast(Callable[[int, int, int], Fraction], _estimate)
    operation = GridPotentialEnergy.make_on(
        grid_points=1,
        system=system,
        workspace=workspace,
        radians_per_unit=theta,
        precision=precision,
        iterations=iterations,
        cubic=cubic,
        adjoint=adjoint,
    )
    gates = DynamicCircuit(logical + work + 1).append(operation).to_ir().instructions
    if electrons == 2:
        pair = GridPairPotential.make_on(
            grid_points=1,
            system_i=system[0],
            system_j=system[1],
            workspace=workspace,
            radians_per_unit=theta,
            precision=precision,
            iterations=iterations,
            cubic=cubic,
            adjoint=adjoint,
        )
        assert (
            DynamicCircuit(logical + work + 1).append(pair).to_ir().instructions
            == gates
        )
    inputs = []
    phases = []
    for word, spectator in product(range(1 << logical), (0, 1)):
        bits = tuple(word >> bit & 1 for bit in reversed(range(logical)))
        points = tuple(bits[3 * e : 3 * e + 3] for e in range(electrons))
        value = Fraction(0)
        for left, right in combinations(points, 2):
            distance = sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
            value += (
                Fraction(cubic_word(distance, precision), 1 << precision)
                if cubic
                else estimate(distance, precision, iterations)
            )
        inputs.append(bits + (0,) * work + (spectator,))
        phases.append(exp((-1 if adjoint else 1) * 1j * theta * float(value)))
    observed, actual_phases, events = certified_and_monomial_map(gates, tuple(inputs))
    assert observed == tuple(inputs)
    torch.testing.assert_close(
        actual_phases, torch.tensor(phases, dtype=torch.complex128), atol=5e-12, rtol=0
    )
    assert events == sum(g.name == "measure" for g in gates) and events > 0
    # Every erased conjunction is checked against actual primitive operands.
    # Positive local branch scalars compose to 2**(-events/2) for every history;
    # complete complex diagonal columns extend to arbitrary coherent inputs.
