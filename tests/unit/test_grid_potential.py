"""Verify the executed Coulomb phases and clean storage independently."""

from cmath import exp
from fractions import Fraction
from itertools import combinations, product
from math import floor, sqrt

import pytest

from flagquantum.algorithms.primitives.grid_potential import (
    GridPotentialEnergy,
    grid_potential_requirements,
)
from flagquantum.circuit import Circuit
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def _estimate(distance, precision, iterations):
    if not distance:
        return Fraction(0)
    scale = 1 << precision
    exponent = (1 - distance.bit_length()) // 2
    power = Fraction(2) ** exponent
    value = Fraction(floor(power * (3 - distance * power**2) * scale / 2), scale)
    for _ in range(iterations):
        value = Fraction(floor(value * (3 - distance * value**2) * scale / 2), scale)
    return value


@pytest.mark.parametrize(
    "electrons,iterations,adjoint", [(2, 0, False), (2, 1, True), (3, 1, False)]
)
def test_grid_potential_complex_state_cleanup(electrons, iterations, adjoint):
    precision, theta = 4, 0.721
    width = 3 * electrons
    work, slots = grid_potential_requirements(
        1, precision=precision, iterations=iterations
    )
    system = tuple(tuple((i * 3 + j,) for j in range(3)) for i in range(electrons))
    op = GridPotentialEnergy.make_on(
        grid_points=1,
        system=system,
        workspace=tuple(range(width, width + work)),
        radians_per_unit=theta,
        precision=precision,
        iterations=iterations,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(width + work).append(op)
    ir = circuit.to_ir()
    initial, expected = {}, {}
    rows = (
        list(product((0, 1), repeat=width))
        if electrons == 2
        else [
            tuple((i >> bit) & 1 for bit in reversed(range(width)))
            for i in (0, 8, 13, 17, 42, 110, 317, 511)
        ]
    )
    for row in rows:
        points = [row[i : i + 3] for i in range(0, width, 3)]
        phase = sum(
            _estimate(
                sum((a - b) ** 2 for a, b in zip(left, right, strict=True)),
                precision,
                iterations,
            )
            for left, right in combinations(points, 2)
        )
        label = sum(bit << i for i, bit in enumerate(reversed(row)))
        index = label << work
        initial[index] = exp(0.347j * label) / sqrt(len(rows))
        expected[index] = initial[index] * exp(
            (-1 if adjoint else 1) * 1j * theta * float(phase)
        )
    for seed in (3, 29):
        observed = execute_sparse_register(ir, initial, seed=seed)
        assert all(
            abs(observed.get(i, 0) - expected.get(i, 0)) < 5e-12
            for i in observed.keys() | expected.keys()
        )

    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(width + work).append(op)
    occupied = DynamicCircuit(width + work + 1)
    occupied.measure(width + work, classical_bit=3)
    occupied.append(op)
    assert {
        instruction.metadata["classical_bit"]
        for instruction in occupied.to_ir().instructions
        if instruction.name == "measure"
    } <= {3, *range(4, 4 + slots)}


def test_one_electron_potential_is_identity():
    circuit = Circuit(3)
    op = GridPotentialEnergy.make_on(
        grid_points=1,
        system=(((0,), (1,), (2,)),),
        workspace=(),
        radians_per_unit=0.2,
        precision=4,
    )
    assert circuit.append(op) is circuit
    assert not circuit.to_ir().instructions


@pytest.mark.parametrize("adjoint", [False, True])
def test_table_cubic_grid_potential_absolute_phase(adjoint):
    from tests.unit.test_grid_cubic_inverse_sqrt import cubic_word

    precision, theta = 5, 0.721
    work, _ = grid_potential_requirements(
        1, precision=precision, iterations=1, cubic=True
    )
    op = GridPotentialEnergy.make_on(
        grid_points=1,
        system=(((0,), (1,), (2,)), ((3,), (4,), (5,))),
        workspace=tuple(range(6, 6 + work)),
        radians_per_unit=theta,
        precision=precision,
        cubic=True,
        adjoint=adjoint,
    )
    ir = DynamicCircuit(6 + work).append(op).to_ir()
    incoming, expected = {}, {}
    for i in range(64):
        index = i << work
        squared = ((i >> 3) ^ (i & 7)).bit_count()
        incoming[index] = exp(0.3j * i) / 8
        expected[index] = incoming[index] * exp(
            (-1 if adjoint else 1)
            * 1j
            * theta
            * cubic_word(squared, precision)
            / (1 << precision)
        )
    observed = execute_sparse_register(ir, incoming, seed=11)
    assert all(
        abs(observed.get(i, 0) - expected.get(i, 0)) < 4e-12
        for i in observed.keys() | expected.keys()
    )
