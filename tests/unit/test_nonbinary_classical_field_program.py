"""Classical field structural admission is independent of quantum execution."""

from dataclasses import replace

import pytest
from sympy import Symbol

from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
    logical_classical_register_width,
)
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.test_classical_word_program import read_function
from tests.unit.test_logical_register_types import identity

pytestmark = pytest.mark.unit


def field_type(
    characteristic: int, degree: int, polynomial_degree: int | None = None
) -> RegisterType:
    order = characteristic**degree
    width = (order - 1).bit_length()
    parameters: tuple[tuple[str, int | str], ...] = (
        ("storage", "classical"),
        ("characteristic", characteristic),
        ("degree", degree),
        ("order", order),
    )
    if polynomial_degree is not None:
        parameters = (
            parameters[0],
            ("polynomial_degree", polynomial_degree),
            *parameters[1:],
        )
        width *= polynomial_degree + 1
    return RegisterType(
        "finite_field" if polynomial_degree is None else "finite_field_polynomial",
        0,
        width,
        parameters,
    )


@pytest.mark.parametrize(
    "characteristic,degree", [(3, 1), (5, 1), (3, 2), (7, 2), (11, 1)]
)
@pytest.mark.parametrize("polynomial_degree", [None, 0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_nonbinary_identity_and_read_keep_nominal_linear_words(
    characteristic: int,
    degree: int,
    polynomial_degree: int | None,
    shape: tuple[int, ...],
) -> None:
    dtype = field_type(characteristic, degree, polynomial_degree)
    assert logical_classical_register_width(dtype) == dtype.classical_bits
    assert executable_classical_register_width(dtype) == dtype.classical_bits
    assert verify_program_record(identity(dtype, shape=shape)) == ()
    assert verify_program_record(ProgramRecord((read_function(dtype),))) == ()
    assert verify_program_record(ProgramRecord((read_function(dtype, delta=1),)))
    with pytest.raises(ValueError):
        executable_register_width(dtype)
    # Proposal 125 admits the distinct raw quantum view. The classical
    # descriptor itself above still cannot enter a quantum binding.
    assert (
        executable_register_width(
            replace(
                dtype,
                quantum_bits=dtype.classical_bits,
                classical_bits=0,
                parameters=dtype.parameters[1:],
            )
        )
        == dtype.classical_bits
    )


@pytest.mark.parametrize(
    "dtype",
    [
        replace(field_type(3, 1), classical_bits=1),
        replace(field_type(3, 2), classical_bits=True),
        replace(field_type(3, 1), quantum_bits=1),
        replace(
            field_type(3, 1),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 1),
                ("degree", 1),
                ("order", 1),
            ),
        ),
        replace(
            field_type(3, 1),
            parameters=(
                ("storage", "classical"),
                ("characteristic", True),
                ("degree", 1),
                ("order", 3),
            ),
        ),
        replace(
            field_type(3, 1),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 3),
                ("degree", 0),
                ("order", 3),
            ),
        ),
        replace(
            field_type(3, 1),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 3),
                ("degree", 1),
                ("order", 4),
            ),
        ),
        replace(
            field_type(3, 1),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 3),
                ("degree", Symbol("d")),
                ("order", 3),
            ),
        ),
        replace(field_type(3, 1), parameters=field_type(3, 1).parameters[::-1]),
        replace(
            field_type(3, 1), parameters=field_type(3, 1).parameters + (("order", 3),)
        ),
        replace(field_type(3, 1, 1), classical_bits=3),
        replace(
            field_type(3, 1, 1),
            parameters=(
                ("storage", "classical"),
                ("polynomial_degree", -1),
                *field_type(3, 1).parameters[1:],
            ),
        ),
    ],
)
def test_malformed_nonbinary_field_metadata_is_refused(dtype: RegisterType) -> None:
    with pytest.raises(ValueError):
        executable_classical_register_width(dtype)
    assert verify_program_record(identity(dtype))
