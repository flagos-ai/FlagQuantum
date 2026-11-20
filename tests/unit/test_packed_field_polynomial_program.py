"""Canonical binary-polynomial ports reuse Program's typed linear verifier."""

from dataclasses import replace

import pytest
from sympy import Symbol

from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
    logical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.test_logical_register_types import identity

pytestmark = pytest.mark.unit


def polynomial_type(degree: int = 2, width: int = 2) -> RegisterType:
    return RegisterType(
        "finite_field_polynomial",
        (degree + 1) * width,
        parameters=(
            ("polynomial_degree", degree),
            ("characteristic", 2),
            ("degree", width),
            ("order", 1 << width),
        ),
    )


@pytest.mark.parametrize("degree,width", [(0, 1), (0, 8), (2, 2), (4, 3)])
@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_canonical_scalar_and_shaped_identity(
    degree: int, width: int, shape: tuple[int, ...]
) -> None:
    dtype = polynomial_type(degree, width)
    assert logical_register_width(dtype) == (degree + 1) * width
    assert executable_register_width(dtype) == (degree + 1) * width
    assert verify_program_record(identity(dtype, shape=shape)) == ()
    with pytest.raises(ValueError):
        executable_classical_register_width(dtype)


BASE = polynomial_type()
INVALID = (
    replace(BASE, quantum_bits=5),
    replace(BASE, quantum_bits=True),
    replace(BASE, classical_bits=1),
    replace(BASE, classical_bits=False),
    replace(BASE, parameters=()),
    replace(BASE, parameters=BASE.parameters[1:]),
    replace(BASE, parameters=BASE.parameters[::-1]),
    replace(BASE, parameters=BASE.parameters + (("degree", 2),)),
    replace(BASE, parameters=(("polynomial_degree", -1),) + BASE.parameters[1:]),
    replace(BASE, parameters=(("polynomial_degree", True),) + BASE.parameters[1:]),
    replace(
        BASE, parameters=(("polynomial_degree", Symbol("n")),) + BASE.parameters[1:]
    ),
    replace(
        BASE,
        parameters=BASE.parameters[:1] + (("characteristic", 3),) + BASE.parameters[2:],
    ),
    replace(BASE, parameters=BASE.parameters[:-1] + (("order", 8),)),
    replace(BASE, parameters=BASE.parameters + (("polynomial", 7),)),
    replace(
        BASE,
        quantum_bits=0,
        classical_bits=6,
        parameters=(("storage", "classical"),) + BASE.parameters,
    ),
)


@pytest.mark.parametrize("dtype", INVALID)
def test_invalid_metadata_fails_before_program_admission(dtype: RegisterType) -> None:
    with pytest.raises(ValueError):
        executable_register_width(dtype)
    with pytest.raises(ValueError):
        logical_register_width(dtype)
    assert verify_program_record(identity(dtype))


def test_split_join_preserves_nominal_type_and_linear_handles() -> None:
    dtype = polynomial_type()
    before, after = ValueRef("entry", 0, "register"), ValueRef("entry", 7, "register")
    bits = tuple(ValueRef("entry", i + 1, "qubit") for i in range(6))
    block = BlockRecord(
        "entry",
        (before,),
        operations=(
            Operation("split_register", (before,), bits),
            Operation("join_register", bits, (after,), register_type=dtype),
        ),
        return_values=(after,),
    )
    function = FunctionRecord("main", (block,), registers=(Register("word", dtype),))
    assert verify_program_record(ProgramRecord((function,))) == ()
    stale = replace(function, blocks=(replace(block, return_values=(before,)),))
    assert "value.linear_duplicate" in {
        d.code for d in verify_program_record(ProgramRecord((stale,)))
    }
    other = polynomial_type(1, 3)
    wrong_join = replace(block.operations[1], register_type=other)
    wrong = replace(
        function, blocks=(replace(block, operations=(block.operations[0], wrong_join)),)
    )
    assert verify_program_record(ProgramRecord((wrong,)))
