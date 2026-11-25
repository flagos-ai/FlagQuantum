"""Approved fixed-point storage preserves canonical and symbolic boundaries."""

from dataclasses import replace

import pytest

from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import RegisterType

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("fraction", [-10000, -7, -2, -1, 0, 1])
def test_canonical_bit_storage(fraction: int) -> None:
    dtype = fixed_point_register_type(1, fraction)
    assert executable_register_width(dtype) == 1
    assert dtype.parameters == (("fractional_width", fraction), ("signed", False))


@pytest.mark.parametrize(
    "dtype",
    [
        fixed_point_register_type(0, -2, True),
        fixed_point_register_type(-1, -2),
        RegisterType(
            "fixed_point", True, parameters=(("fractional_width", 1), ("signed", False))
        ),
        replace(fixed_point_register_type(1, 1), classical_bits=True),
        replace(fixed_point_register_type(1, 1), classical_bits=1),
        RegisterType(
            "fixed_point", 1, parameters=(("signed", False), ("fractional_width", 1))
        ),
        RegisterType(
            "fixed_point", 1, parameters=(("fractional_width", True), ("signed", False))
        ),
        RegisterType(
            "fixed_point", 1, parameters=(("fractional_width", 1), ("signed", 0))
        ),
        RegisterType(
            "fixed_point", 1, parameters=(("fractional_width", 1), ("signed", True))
        ),
        RegisterType(
            "fixed_point", 1, parameters=(("fractional_width", 2), ("signed", False))
        ),
        RegisterType("fixed_point", 1),
    ],
)
def test_other_storage_and_noncanonical_records_are_rejected(
    dtype: RegisterType,
) -> None:
    with pytest.raises(ValueError):
        executable_register_width(dtype)


def test_symbolic_storage_retains_physical_refusal() -> None:
    import sympy

    width = sympy.Symbol("n", integer=True, positive=True)
    for dtype in (
        fixed_point_register_type(width, 1),
        fixed_point_register_type(1, width),
    ):
        with pytest.raises(ValueError):
            executable_register_width(dtype)
