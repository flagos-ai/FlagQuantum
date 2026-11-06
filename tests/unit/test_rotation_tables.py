"""Exact fixed-point words from the locked rotation tree."""

from itertools import product
from math import cos, pi, sin
from random import Random

import pytest

from flagquantum.algorithms.primitives.rotation_tables import rotation_tables

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [2, 3, 8, 12])
def test_original_rotation_words(width: int) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        RotationTree,
    )

    rng = Random(345)
    vectors = [(1j,), (0.5, -0.5, 0.5j, -0.5j), (0, 0, 1j, 0)]
    for size in (2, 4, 8, 16):
        for _ in range(20):
            values = tuple(
                complex(rng.randrange(-5, 6), rng.randrange(-5, 6)) for _ in range(size)
            )
            norm = sum(abs(value) ** 2 for value in values) ** 0.5
            if norm:
                vectors.append(tuple(value / norm for value in values))

    for word in range(1 << (width - 1)):
        half_angle = pi * (word + 0.5) / (1 << width)
        vectors.append((cos(half_angle), sin(half_angle)))

    for values, uncompute in product(vectors, (False, True)):
        rows, phases = RotationTree(np.asarray(values), width, uncompute).get_rom_vals()
        assert rotation_tables(values, width, uncompute=uncompute) == (
            tuple(map(tuple, rows)),
            tuple(phases),
        )


@pytest.mark.parametrize(
    "values,width", [((), 2), ((1, 0, 0), 2), ((2,), 2), ((1,), 1)]
)
def test_invalid_tables(values: tuple[complex, ...], width: int) -> None:
    with pytest.raises(ValueError):
        rotation_tables(values, width)
