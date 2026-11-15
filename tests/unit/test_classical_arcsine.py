"""Original ArcSin scalar semantics and source encoding edge cases."""

from math import sqrt

import pytest
from sympy import Symbol

from flagquantum.simulation.classical_arcsine import source_arcsine_classical_values

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 10])
def test_locked_callback_whole_scalar_domain(width: int) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin

    for fraction in {0, width // 2, width}:
        original = ArcSin(width, fraction)
        for x in range(1 << width):
            occupied = ((1 << width) - 1) if x & 1 else 0
            expected = original.on_classical_vals(x=x, result=occupied)
            assert source_arcsine_classical_values(x, occupied, width) == (
                expected["x"],
                expected["result"],
            )


def test_original_example_and_unmasked_output() -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin

    x = int(1024 / sqrt(2))
    assert source_arcsine_classical_values(x, 0, 10) == ArcSin(10, 7).call_classically(
        x=x, result=0
    )
    assert source_arcsine_classical_values(15, 0, 4) == (15, 19)


@pytest.mark.parametrize("x,width", [(-1, 4), (16, 4), ((1 << 60) - 1, 60)])
def test_original_floating_domain_assertion(x: int, width: int) -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin

    with pytest.raises(AssertionError):
        ArcSin(width, width).on_classical_vals(x=x, result=0)
    with pytest.raises(AssertionError):
        source_arcsine_classical_values(x, 0, width)


def test_symbolic_width_is_an_estimate_only() -> None:
    from qualtran.bloqs.arithmetic.trigonometric import ArcSin

    n = Symbol("n")
    with pytest.raises(ValueError, match="Symbolic bitsize n"):
        ArcSin(n, n).on_classical_vals(x=0, result=0)
    with pytest.raises(ValueError, match="Symbolic bitsize n"):
        source_arcsine_classical_values(0, 0, n)
