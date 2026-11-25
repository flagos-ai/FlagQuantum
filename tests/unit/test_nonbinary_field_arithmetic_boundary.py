"""Raw field storage approval does not enable binary arithmetic or controls."""

import pytest

from flagquantum.compiler.registers.control.component_predicate import (
    validate_control_ports,
)
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
    constant_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.finite_field.inverse_component import (
    field_inverse_program,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    field_polynomial_addition_program,
)
from flagquantum.compiler.registers.finite_field.product_erasure_program import (
    product_erasure_program,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("characteristic,degree", ((3, 1), (5, 1), (3, 2)))
@pytest.mark.parametrize(
    "operation",
    (
        "add",
        "add_constant",
        "frobenius",
        "scale",
        "inverse",
        "product",
        "erase",
        "poly_add",
        "poly_constant",
    ),
)
def test_nonbinary_fields_keep_existing_binary_arithmetic_refusal(
    characteristic: int, degree: int, operation: str
) -> None:
    dtype = FiniteFieldEncoding(characteristic, degree).register_type()
    width = dtype.num_qubits
    assert isinstance(width, int)
    modulus = (1 << width) | 3
    with pytest.raises(ValueError, match="binary|unsupported executable"):
        if operation == "add":
            field_addition_program(dtype)
        elif operation == "add_constant":
            field_addition_program(dtype, constant=1)
        elif operation == "frobenius":
            field_frobenius_program(dtype, modulus=modulus)
        elif operation == "scale":
            constant_multiplication_program(dtype, 1, modulus=modulus)
        elif operation == "inverse":
            field_inverse_program(dtype, modulus=modulus)
        elif operation == "product":
            karatsuba_multiplication_program(dtype, modulus=modulus)
        elif operation == "erase":
            product_erasure_program(dtype)
        else:
            field_polynomial_addition_program(
                FieldPolynomialLayout(1, width),
                dtype,
                constants=(1,) if operation == "poly_constant" else None,
            )


@pytest.mark.parametrize("characteristic,degree", ((3, 1), (5, 1), (3, 2)))
@pytest.mark.parametrize("polynomial", (False, True))
@pytest.mark.parametrize("shape", ((), (0,)))
def test_nonbinary_controls_remain_refused_even_in_empty_shaped_groups(
    characteristic: int, degree: int, polynomial: bool, shape: tuple[int, ...]
) -> None:
    field = FiniteFieldEncoding(characteristic, degree)
    dtype = (
        FiniteFieldPolynomialEncoding(1, field).register_type()
        if polynomial
        else field.register_type()
    )
    with pytest.raises(ValueError, match="unsupported executable"):
        validate_control_ports(
            (Register("field", dtype, shape), Register("q", quantum_bit_type())), None
        )
