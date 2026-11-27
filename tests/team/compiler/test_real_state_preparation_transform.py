"""Walsh ordering and absolute phase of the native preparation ladder."""

import math
import random

import pytest
import torch

from flagquantum.compiler.optimization.state_preparation_synthesis import (
    _synthesize_real_state,
    _walsh,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("controls", range(9))
def test_transform_matches_independent_character_sums(controls: int) -> None:
    rng = random.Random(47 + controls)
    angles = tuple(rng.uniform(-math.pi, math.pi) for _ in range(1 << controls))
    expected = [
        math.fsum(
            (-angle if (index & branch).bit_count() % 2 else angle)
            for branch, angle in enumerate(angles)
        )
        / len(angles)
        for index in range(len(angles))
    ]

    actual = _walsh(angles, controls)

    assert actual == pytest.approx(expected, rel=0, abs=1e-14)
    assert _walsh(actual, controls) == pytest.approx(
        [angle / len(angles) for angle in angles], rel=0, abs=1e-14
    )


def test_large_character_preserves_branch_order_and_exact_zeroes() -> None:
    controls = 13
    character = 5273
    angles = [
        -0.5 if (character & branch).bit_count() % 2 else 0.5
        for branch in range(1 << controls)
    ]
    original = angles.copy()

    actual = _walsh(angles, controls)

    assert actual[character] == 0.5
    assert all(angle == 0.0 for index, angle in enumerate(actual) if index != character)
    assert angles == original


@pytest.mark.parametrize("width", (1, 3, 8))
@pytest.mark.parametrize("sign", (-1, 1))
def test_signed_state_retains_absolute_phase_and_permuted_ports(
    width: int, sign: int
) -> None:
    rng = random.Random(51 + width)
    vector = [rng.uniform(-1, 1) for _ in range(1 << width)]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    vector = [sign * value / norm for value in vector]
    wires = tuple(reversed(range(1, width + 1)))
    n_wires = width + 2
    instructions = _synthesize_real_state(vector, wires)
    program = CircuitIR(n_wires=n_wires, instructions=instructions, dtype="complex128")
    program.validate()

    actual = run_local_statevector(
        program, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    ).reshape(-1)
    expected = torch.zeros(1 << n_wires, dtype=torch.complex128)
    for word, amplitude in enumerate(vector):
        physical = sum(
            ((word >> (width - 1 - index)) & 1) << (n_wires - 1 - wire)
            for index, wire in enumerate(wires)
        )
        expected[physical] = amplitude
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert {wire for gate in instructions for wire in gate.wires}.issubset(wires)
