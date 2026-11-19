"""Core validates declared signatures and caller ownership without invented bodies."""

from dataclasses import replace

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

pytestmark = pytest.mark.unit


def declared_program(*, directional: bool = False) -> ProgramRecord:
    dtype = RegisterType("uint", 3)
    ports = (
        (
            Register("input", dtype, side=RegisterSide.LEFT),
            Register("output", dtype, side=RegisterSide.RIGHT),
        )
        if directional
        else (Register("word", dtype),)
    )
    q, r = ValueRef("entry", 0, "register"), ValueRef("entry", 1, "register")
    root = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(Operation("call", (q,), (r,), callee="declared"),),
                return_values=(r,),
            ),
        ),
        registers=ports,
    )
    return ProgramRecord((root, FunctionRecord("declared", (), registers=ports)))


@pytest.mark.parametrize("directional", [False, True])
def test_declared_call_preserves_fresh_ownership_and_refuses_payload_spoofing(
    directional: bool,
) -> None:
    record = declared_program(directional=directional)
    assert verify_program_record(record) == ()
    assert program_to_circuit(record).state == "unsupported_with_diagnostics"
    spoofed = replace(record, canonical_payload=CircuitIR(3, ()).to_dict())
    assert program_to_circuit(spoofed).state == "unsupported_with_diagnostics"
    assert record.functions[1] != replace(
        record.functions[1], blocks=(BlockRecord("entry", return_values=()),)
    )


@pytest.mark.parametrize(
    "defect",
    [
        "untyped",
        "entry",
        "dtype",
        "shape",
        "parameter",
        "ports",
        "stale",
        "arity",
        "kind",
        "unreachable",
    ],
)
def test_malformed_declarations_and_calls_are_rejected(defect: str) -> None:
    record = declared_program()
    main, leaf = record.functions
    block = main.blocks[0]
    op = block.operations[0]
    if defect == "untyped":
        leaf = replace(leaf, registers=None)
    elif defect == "entry":
        leaf = replace(leaf, entry="missing")
    elif defect in {"dtype", "shape", "parameter", "ports"}:
        reg = Register("word", RegisterType("uint", 3))
        if defect == "dtype":
            reg = Register(reg.name, RegisterType("uint", -1))
        elif defect == "shape":
            reg = Register("word", reg.dtype, (-1,))
        elif defect == "parameter":
            reg = Register(reg.name, replace(reg.dtype, parameters=(("unknown", 1),)))
        else:
            reg = Register("word", RegisterType("uint", 4))
        leaf = replace(leaf, registers=(reg,))
    elif defect == "stale":
        main = replace(main, blocks=(replace(block, return_values=block.arguments),))
    elif defect == "arity":
        main = replace(
            main, blocks=(replace(block, operations=(replace(op, operands=()),)),)
        )
    elif defect == "kind":
        main = replace(
            main,
            blocks=(
                replace(
                    block,
                    operations=(
                        replace(op, results=(replace(op.results[0], kind="qubit"),)),
                    ),
                ),
            ),
        )
    else:
        leaf = replace(leaf, name="unreachable", entry="invalid")
        main = replace(
            main, blocks=(replace(block, operations=(), return_values=block.arguments),)
        )
    assert verify_program_record(replace(record, functions=(main, leaf)))
