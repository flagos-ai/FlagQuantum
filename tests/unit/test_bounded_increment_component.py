"""Typed bounded increment retains complete control, padding and phase action."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.bounded_increment_component import (
    bounded_increment_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,modulus", [(1, 2), (2, 3), (2, 4)])
@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
@pytest.mark.parametrize("adjoint", [False, True])
def test_atomic_increment_full_operator_padding_controls_and_clean_work(
    width: int, modulus: int, values: tuple[int, ...], adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic import ModAddK

    from flagquantum.compiler.registers.arithmetic.bounded_increment import (
        bounded_increment_requirements,
    )

    source = ModAddK(width, modulus, -1 if adjoint else 1, cvs=values)
    graph = bounded_increment_program(width, modulus, values, adjoint=adjoint)
    logical = width + len(values)
    work, _ = bounded_increment_requirements(width, modulus, len(values))
    bound = bind_register_program(
        graph,
        ((0,), (1,), tuple(range(2, logical))),
        clean=tuple(range(logical, logical + work)),
    )
    actual = logical_operator(bound.circuit, logical)
    expected = torch.zeros_like(actual)
    words = 1 << width
    for ctrl, value in product(range(4), range(words)):
        output = source.on_classical_vals(ctrl=ctrl, x=value)
        expected[ctrl * words + output["x"], ctrl * words + value] = 1
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=0)
