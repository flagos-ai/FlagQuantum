"""Finite gate calls preserve shared SSA, ownership and payload attestation."""

from dataclasses import replace

import pytest

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    program_to_circuit,
    verify_program_record,
)
from tests.unit.program_call_fixtures import wrap_circuit

pytestmark = pytest.mark.unit


def codes(record: ProgramRecord) -> set[str]:
    return {item.code for item in verify_program_record(record)}


def test_explicit_and_implicit_return_hashes_and_identity() -> None:
    block = BlockRecord("entry")
    assert hash(block) == hash(("entry", (), "return", (), ()))
    explicit = replace(block, return_values=())
    assert explicit != block
    assert hash(explicit) == hash(("entry", (), "return", (), (), ()))
    operation = Operation("x")
    old = ("x", (), (), (), None, ())
    assert hash(operation) == hash(old)
    call = Operation("call", callee="leaf")
    assert hash(call) == hash(("call", (), (), (), None, (), "leaf"))


def test_nested_calls_are_verified_but_require_expansion_for_restoration() -> None:
    circuit = CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))
    record = wrap_circuit(circuit, 3)
    assert codes(record) == set()
    assert program_to_circuit(record).state == "unsupported_with_diagnostics"
    leaf = record.functions[-1].blocks[0]
    flat = replace(record, functions=(FunctionRecord("main", (leaf,)),))
    assert program_to_circuit(flat).canonical_payload == circuit.to_dict()
    assert leaf.return_values is not None
    permuted = replace(leaf, return_values=tuple(reversed(leaf.return_values)))
    tampered = replace(flat, functions=(FunctionRecord("main", (permuted,)),))
    assert codes(tampered) == set()
    assert program_to_circuit(tampered).state == "unsupported_with_diagnostics"


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "cycle",
        "unreachable_cycle",
        "implicit",
        "branch",
        "literal",
        "parameter",
        "condition",
        "primitive_callee",
        "bad_kind",
        "negative",
        "redefine_kind",
    ],
)
def test_invalid_call_profiles_fail_closed(defect: str) -> None:
    q = ValueRef("entry", 0, "qubit")
    r = ValueRef("entry", 1, "qubit")
    leaf = FunctionRecord("leaf", (BlockRecord("entry", (q,), return_values=(q,)),))
    operation = Operation("call", (q,), (r,), callee="leaf")
    main = FunctionRecord(
        "main",
        (BlockRecord("entry", (q,), operations=(operation,), return_values=(r,)),),
    )
    extra: tuple[FunctionRecord, ...] = ()
    if defect == "missing":
        operation = replace(operation, callee="absent")
    elif defect == "duplicate":
        extra = (leaf,)
    elif defect in {"cycle", "unreachable_cycle"}:
        name = "main" if defect == "cycle" else "cycle"
        cyc = FunctionRecord(
            name,
            (
                BlockRecord(
                    "entry",
                    (q,),
                    operations=(Operation("call", (q,), (r,), callee=name),),
                    return_values=(r,),
                ),
            ),
        )
        if defect == "cycle":
            main = cyc
            operation = cyc.blocks[0].operations[0]
        else:
            extra = (cyc,)
    elif defect == "implicit":
        leaf = replace(leaf, blocks=(replace(leaf.blocks[0], return_values=None),))
    elif defect == "branch":
        leaf = replace(leaf, blocks=(replace(leaf.blocks[0], terminator="branch"),))
    elif defect == "literal":
        operation = replace(operation, literal=0.0)
    elif defect == "parameter":
        operation = replace(operation, parameters=(("theta", q),))
    elif defect == "condition":
        operation = replace(operation, condition_clauses=((ConditionRef(q, True),),))
    elif defect == "primitive_callee":
        operation = replace(operation, opcode="x")
    elif defect == "bad_kind":
        operation = replace(operation, results=(replace(r, kind="scalar"),))
    elif defect == "negative":
        operation = replace(operation, results=(replace(r, index=-1),))
    else:
        operation = replace(operation, results=(replace(q, kind="scalar"),))
    main = replace(main, blocks=(replace(main.blocks[0], operations=(operation,)),))
    assert codes(ProgramRecord((main, leaf, *extra)))


@pytest.mark.parametrize("returns", [(), None, "stale", "unknown", "alias"])
def test_return_ownership_includes_all_live_qubits(
    returns: tuple[()] | str | None,
) -> None:
    q = ValueRef("entry", 0, "qubit")
    r = ValueRef("entry", 1, "qubit")
    named_returns = {
        "stale": (q,),
        "unknown": (ValueRef("entry", 9, "qubit"),),
        "alias": (r, r),
    }
    value = named_returns[returns] if isinstance(returns, str) else returns
    block = BlockRecord(
        "entry", (q,), operations=(Operation("x", (q,), (r,)),), return_values=value
    )
    # Introduce a callee so missing explicit returns selects the finite-call profile.
    extra = FunctionRecord("leaf", (BlockRecord("entry", return_values=()),))
    assert codes(ProgramRecord((FunctionRecord("main", (block,)), extra)))


def test_classical_returns_fan_out_and_zero_qubit_functions_work() -> None:
    value = ValueRef("entry", 0, "bool")
    leaf = FunctionRecord(
        "leaf", (BlockRecord("entry", (value,), return_values=(value, value)),)
    )
    a, b = (ValueRef("entry", i, "bool") for i in (1, 2))
    main = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (value,),
                operations=(Operation("call", (value,), (a, b), callee="leaf"),),
                return_values=(a, b),
            ),
        ),
    )
    assert codes(ProgramRecord((main, leaf))) == set()


def test_physical_profiles_cannot_hide_finite_call_fields() -> None:
    from flagquantum.core.ir.program.physical import IsingContext

    module = wrap_circuit(CircuitIR(1, ()))
    context = IsingContext(
        ("a", "b", "c", "d"), "1", "left_associated", "nayak_ising", "1"
    )
    assert "program.call_structure" in codes(replace(module, physical_context=context))


@pytest.mark.parametrize(
    "missing_operand,missing_result", [(True, False), (False, True)]
)
def test_resolved_callee_arity_is_checked(
    missing_operand: bool, missing_result: bool
) -> None:
    module = wrap_circuit(CircuitIR(1, ()))
    main, leaf = module.functions
    operation = main.blocks[0].operations[0]
    operation = replace(
        operation,
        operands=() if missing_operand else operation.operands,
        results=() if missing_result else operation.results,
    )
    main = replace(main, blocks=(replace(main.blocks[0], operations=(operation,)),))
    expected = "program.operand_arity" if missing_operand else "program.result_kind"
    assert expected in codes(replace(module, functions=(main, leaf)))


@pytest.mark.parametrize("field", ["callee", "return_values"])
def test_physical_source_schema_stays_separate_from_internal_call_fields(
    field: str,
) -> None:
    from flagquantum.core.ir.program.physical_codec import (
        native_program_from_dict,
        native_program_to_dict,
    )
    from flagquantum.core.ising import IsingBraidProgram
    from flagquantum.errors import SerializationError

    source = native_program_to_dict(IsingBraidProgram().exchange(0).to_ir())
    restored = native_program_from_dict(source)
    assert verify_program_record(restored) == ()
    block = source["functions"][0]["blocks"][0]
    if field == "callee":
        block["operations"][0][field] = None
    else:
        block[field] = None
    with pytest.raises(SerializationError, match="exactly fields"):
        native_program_from_dict(source)
