"""Shared private-work register graph construction for binding checks."""

from dataclasses import replace

from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from tests.unit.program_call_fixtures import explicit_block


def allocated_circuit(circuit: CircuitIR, data_width: int) -> ProgramRecord:
    """Give an actual gate body private clean work and return only its data."""
    leaf = FunctionRecord("gates", (explicit_block(circuit),))
    args = tuple(ValueRef("entry", i, "qubit") for i in range(data_width))
    work = ValueRef("entry", data_width, "register")
    width = circuit.n_wires - data_width
    dtype = RegisterType("opaque", width)
    bits = tuple(ValueRef("entry", data_width + 1 + i, "qubit") for i in range(width))
    out = tuple(
        ValueRef("entry", circuit.n_wires + 1 + i, "qubit")
        for i in range(circuit.n_wires)
    )
    joined = ValueRef("entry", 2 * circuit.n_wires + 1, "register")
    data_word = ValueRef("entry", 10000, "register")
    output_word = ValueRef("entry", 10001, "register")
    data_type = RegisterType("opaque", data_width)
    operations = (
        Operation("split_register", (data_word,), args),
        Operation(
            "allocate_register",
            results=(work,),
            register_type=dtype,
            workspace_dirty=False,
        ),
        Operation("split_register", (work,), bits),
        Operation("call", (*args, *bits), out, callee="gates"),
        Operation("join_register", out[data_width:], (joined,), register_type=dtype),
        Operation("free_register", (joined,), workspace_dirty=False),
        Operation(
            "join_register", out[:data_width], (output_word,), register_type=data_type
        ),
    )
    main = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (data_word,),
                operations=operations,
                return_values=(output_word,),
            ),
        ),
    )
    main = replace(main, registers=(Register("data", data_type),))
    module = ProgramRecord((main, leaf))
    assert verify_program_record(module) == ()
    return module
