"""Verify exact triangular uniform support and measurement cleanup."""

from itertools import product
from math import sqrt

import pytest

from flagquantum.compiler.registers.preparation.triangular_uniform_preparation import (
    triangular_uniform_ir,
    triangular_uniform_requirements,
)
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("extent", [1, 2, 3, 5])
def test_exact_triangular_support_and_inverse(extent):
    width = (extent - 1).bit_length()
    rotation = 2 * width
    work, slots = triangular_uniform_requirements(extent)
    gates = tuple(
        triangular_uniform_ir(
            extent,
            tuple(range(width)),
            tuple(range(width, rotation)),
            rotation,
            tuple(range(rotation + 1, rotation + 1 + work)),
            tuple(range(slots)),
            adjoint=adjoint,
        )
        for adjoint in (False, True)
    )
    expected = {
        ((p << width) | q) << (1 + work): 1 / sqrt(extent * (extent + 1) // 2)
        for q in range(extent)
        for p in range(q + 1)
    }
    for stream, source, target in (
        (gates[0], {0: 1}, expected),
        (gates[1], expected, {0: 1}),
    ):
        events = sum(op.name == "measure" for op in stream)
        histories = (
            product((0, 1), repeat=events)
            if events <= 8
            else (
                (pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0))
            )
        )
        for history in histories:
            assert _branch(
                stream,
                rotation + 1 + work,
                history,
                source,
                normalize_measurements=True,
            ) == pytest.approx(target, abs=2e-12)
