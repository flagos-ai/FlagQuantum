"""Logical Program validity is independent of supported physical encodings."""

from dataclasses import replace

import pytest

from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    fixed_point_register_type,
)
from flagquantum.core.ir.program.register_verifier import (
    executable_register_width,
    logical_register_width,
)
from flagquantum.core.ir.program.registers import Register, RegisterType

pytestmark = [pytest.mark.unit]


def identity(dtype: RegisterType, *, shape: tuple[int, ...] = ()) -> ProgramRecord:
    import itertools

    values = tuple(
        ValueRef("entry", i, "register")
        for i, _ in enumerate(itertools.product(*(range(n) for n in shape)))
    )
    function = FunctionRecord(
        "main",
        (BlockRecord("entry", values, return_values=values),),
        registers=(Register("word", dtype, shape),),
    )
    return ProgramRecord((function,))


@pytest.mark.parametrize(
    "dtype",
    [
        fixed_point_register_type(8, 6, True),
        fixed_point_register_type(8, 8),
        bounded_unsigned_register_type(4, 3),
    ],
)
@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_numerical_metadata_survives_logical_calls(
    dtype: RegisterType, shape: tuple[int, ...]
) -> None:
    record = identity(dtype, shape=shape)
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == (Register("word", dtype, shape),)
    assert logical_register_width(dtype) == dtype.quantum_bits
    assert executable_register_width(dtype) == dtype.quantum_bits


def test_symbolic_widths_keep_finite_ownership_and_reject_symbolic_shapes() -> None:
    sympy = pytest.importorskip("sympy")
    n = sympy.Symbol("n", integer=True, positive=True)
    for dtype in (
        RegisterType("uint", n),
        RegisterType("opaque", n),
        RegisterType("signed", n),
        bounded_unsigned_register_type(n, 3),
        fixed_point_register_type(n, 1),
    ):
        record = identity(dtype, shape=(2,))
        assert verify_program_record(record) == ()
        function = record.functions[0]
        invalid = replace(function, registers=(Register("word", dtype, (n,)),))
        assert {d.code for d in verify_program_record(ProgramRecord((invalid,)))} == {
            "program.register_type"
        }
        with pytest.raises(ValueError, match="unsupported executable"):
            executable_register_width(dtype)


def test_fixed_point_split_join_has_logical_arity_and_preserves_type() -> None:
    dtype = fixed_point_register_type(3, 1, True)
    word = ValueRef("entry", 0, "register")
    bits = tuple(ValueRef("entry", i + 1, "qubit") for i in range(3))
    out = ValueRef("entry", 4, "register")
    block = BlockRecord(
        "entry",
        (word,),
        operations=(
            Operation("split_register", (word,), bits),
            Operation("join_register", bits, (out,), register_type=dtype),
        ),
        return_values=(out,),
    )
    function = FunctionRecord("main", (block,), registers=(Register("word", dtype),))
    assert verify_program_record(ProgramRecord((function,))) == ()
    changed = replace(
        block.operations[1], register_type=fixed_point_register_type(3, 2)
    )
    bad = replace(
        function, blocks=(replace(block, operations=(block.operations[0], changed)),)
    )
    assert "program.register_type" in {
        d.code for d in verify_program_record(ProgramRecord((bad,)))
    }


@pytest.mark.parametrize(
    "dtype",
    [
        RegisterType("alien", 3),
        RegisterType("uint", -1),
        RegisterType("uint", True),
        RegisterType("uint", 3, parameters=(("unknown", 1),)),
        RegisterType("uint", 3, parameters=(("bound", 20),)),
        RegisterType("uint", 3, parameters=(("bound", True),)),
        RegisterType(
            "fixed_point", 3, parameters=(("fractional_width", 4), ("signed", False))
        ),
        RegisterType(
            "fixed_point", 3, parameters=(("fractional_width", 1), ("signed", 1))
        ),
        RegisterType(
            "fixed_point", 3, parameters=(("signed", True), ("fractional_width", 1))
        ),
        RegisterType("bit", 2),
        RegisterType("uint", 3, 1),
    ],
)
def test_bad_logical_descriptors_are_rejected_before_graph_admission(
    dtype: RegisterType,
) -> None:
    with pytest.raises(ValueError):
        logical_register_width(dtype)
    assert "program.register_type" in {
        d.code for d in verify_program_record(identity(dtype))
    }
