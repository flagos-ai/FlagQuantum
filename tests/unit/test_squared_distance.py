"""Verify signed differences in a clean squared-distance XOR."""

from itertools import product

import pytest

from flagquantum.compiler.registers.squared_distance import (
    squared_distance_ir,
    squared_distance_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2])
def test_squared_distance_all_coordinates_and_measurement_branches(width):
    result_width, work_width, slots = squared_distance_requirements(width)
    coords = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(6))
    result = tuple(range(6 * width, 6 * width + result_width))
    work = tuple(range(6 * width + result_width, 6 * width + result_width + work_width))
    gates = squared_distance_ir(
        coords[:3], coords[3:], result, work, tuple(range(slots))
    )

    def word(value, size):
        return tuple((value >> i) & 1 for i in reversed(range(size)))

    before, after = [], []
    for axes in product(range(1 << width), repeat=6):
        distance = sum((a - b) ** 2 for a, b in zip(axes[:3], axes[3:], strict=True))
        position = tuple(bit for a in axes for bit in word(a, width))
        seed = 1 if axes[0] % 2 else 0
        before.append(position + word(seed, result_width) + (0,) * work_width)
        after.append(position + word(seed ^ distance, result_width) + (0,) * work_width)
    events = sum(g.name == "measure" for g in gates)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)
