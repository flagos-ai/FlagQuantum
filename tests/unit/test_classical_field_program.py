"""Classical binary fields reuse canonical validation and owned word reads."""

from dataclasses import replace

import pytest
from sympy import Symbol

from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
)
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.test_classical_word_program import read_function
from tests.unit.test_logical_register_types import identity
from tests.unit.test_packed_field_polynomial_program import INVALID

pytestmark = pytest.mark.unit


def field_type(width: int, polynomial_degree: int | None = None) -> RegisterType:
    parameters: tuple[tuple[str, int | str], ...] = (
        ("storage", "classical"),
        ("characteristic", 2),
        ("degree", width),
        ("order", 1 << width),
    )
    size = width
    if polynomial_degree is not None:
        parameters = (
            parameters[0],
            ("polynomial_degree", polynomial_degree),
            *parameters[1:],
        )
        size *= polynomial_degree + 1
    return RegisterType(
        "finite_field" if polynomial_degree is None else "finite_field_polynomial",
        0,
        size,
        parameters,
    )


@pytest.mark.parametrize(
    "width,degree", [(1, None), (3, None), (8, None), (1, 0), (2, 1), (3, 4)]
)
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_identity_array_read_and_linear_successor(
    width: int, degree: int | None, shape: tuple[int, ...]
) -> None:
    dtype = field_type(width, degree)
    assert executable_classical_register_width(dtype) == dtype.classical_bits
    assert verify_program_record(identity(dtype, shape=shape)) == ()
    function = read_function(dtype)
    assert verify_program_record(ProgramRecord((function,))) == ()
    assert verify_program_record(ProgramRecord((read_function(dtype, delta=1),)))
    block = function.blocks[0]
    stale = replace(function, blocks=(replace(block, return_values=block.arguments),))
    assert "value.linear_duplicate" in {
        d.code for d in verify_program_record(ProgramRecord((stale,)))
    }
    with pytest.raises(ValueError):
        executable_register_width(dtype)


@pytest.mark.parametrize("bad_quantum", INVALID[:-1])
def test_classical_projection_reuses_every_invalid_polynomial_rule(
    bad_quantum: RegisterType,
) -> None:
    dtype = replace(
        bad_quantum,
        quantum_bits=0,
        classical_bits=bad_quantum.quantum_bits,
        parameters=(("storage", "classical"),) + bad_quantum.parameters,
    )
    # An invalid nonzero classical component is not part of a quantum descriptor
    # projection. Keep that malformed mixed-storage example explicitly mixed.
    if bad_quantum.classical_bits != 0 or type(bad_quantum.classical_bits) is not int:
        dtype = replace(dtype, quantum_bits=1)
    with pytest.raises(ValueError):
        executable_classical_register_width(dtype)
    assert verify_program_record(identity(dtype))


@pytest.mark.parametrize(
    "dtype",
    [
        replace(field_type(2), quantum_bits=1),
        replace(field_type(2), classical_bits=True),
        replace(
            field_type(2),
            parameters=(("characteristic", 2), ("degree", 2), ("order", 4)),
        ),
        replace(
            field_type(2),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 3),
                ("degree", 2),
                ("order", 9),
            ),
        ),
        replace(
            field_type(2),
            parameters=(
                ("storage", "classical"),
                ("characteristic", 2),
                ("degree", Symbol("n")),
                ("order", 4),
            ),
        ),
        replace(field_type(2), parameters=field_type(2).parameters[::-1]),
        replace(field_type(2), parameters=field_type(2).parameters + (("order", 4),)),
    ],
)
def test_invalid_field_storage_is_refused(dtype: RegisterType) -> None:
    with pytest.raises(ValueError):
        executable_classical_register_width(dtype)
    assert verify_program_record(identity(dtype))


def test_same_width_nominal_field_return_mismatch_is_refused() -> None:
    dtype = field_type(3)
    parameters = dtype.parameters + (
        ("polynomial", 11),
        ("polynomial_field_characteristic", 2),
        ("polynomial_field_degree", 1),
        ("polynomial_field_primitive_element", 1),
        ("polynomial_field_modulus", 3),
    )
    first = replace(dtype, parameters=parameters)
    second = replace(
        first, parameters=parameters[:-5] + (("polynomial", 13),) + parameters[-4:]
    )
    assert (
        executable_classical_register_width(first)
        == executable_classical_register_width(second)
        == 3
    )
    function = read_function(first)
    from flagquantum.core.ir.program.registers import Register

    wrong = replace(function, registers=(Register("a", second),))
    # Reads preserve the input nominal identity through the existing successor.
    # A callee returning a different nominal identity fails at call validation.
    from flagquantum.core.ir.program import (
        BlockRecord,
        FunctionRecord,
        Operation,
        ValueRef,
    )

    before, after = ValueRef("entry", 0, "register"), ValueRef("entry", 1, "register")
    root = FunctionRecord(
        "root",
        (
            BlockRecord(
                "entry",
                (before,),
                operations=(Operation("call", (before,), (after,), callee="main"),),
                return_values=(after,),
            ),
        ),
        registers=(Register("a", first),),
    )
    assert verify_program_record(ProgramRecord((root, wrong)))
