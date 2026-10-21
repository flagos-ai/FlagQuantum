"""Conformance suite for the internal multi-level IR boundary.

`contracts/multi-level-ir-internal-v1-candidate.json` declares claims about
the boundary between the public IR and the internal program, quantum, and target
levels: level entry and exit, value identity, linearity, joins, failure
categories, round trip, dynamic def-use, and rejection.
`flagquantum.core.ir.levels` supplies the
vocabulary those claims are spelled in, plus the contract fake.

This module is the conformance suite, not a test of the fake's internals. The
fake passes it today; the real program, quantum, and target levels are required
to pass the same suite before they replace it, which is the replacement
demonstration engineering decision principle 10 asks for. Every test here reads
the boundary through its declared entry points, so a second implementation can
be substituted without touching this file.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.core.ir import levels as level_boundary
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "multi-level-ir-internal-v1-candidate.json"
PUBLIC_SNAPSHOT = ROOT / "docs" / "public_api_v1.json"
GATE = ROOT / "tools" / "check_multi_level_ir_contract.py"

_SPEC = importlib.util.spec_from_file_location("check_multi_level_ir_contract", GATE)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _static_circuit() -> CircuitIR:
    """Return a circuit that exercises every canonical payload section."""
    return CircuitIR(
        n_wires=3,
        instructions=(
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rz", (2,), {"theta": 0.375}),
            Instruction("swap", (1, 2)),
        ),
        observables=(ObservableNode("z", (0,), coefficient=0.5),),
        measurements=(MeasurementNode("probability", (0, 1)),),
        metadata={"seed": 7, "origin": "conformance"},
    )


def _dynamic_circuit() -> CircuitIR:
    """Return measurement, reset, overwrite, and DNF through the public builder."""

    circuit = DynamicCircuit(3)
    circuit.h(0)
    circuit.measure(0, classical_bit=2)
    circuit.reset(0)
    circuit.measure(1, classical_bit=0)
    circuit.conditional("x", 2, any_of=({0: 0}, {2: 1}))
    return circuit.to_ir()


def _record_with(**overrides: object) -> level_boundary.ProgramRecord:
    """Return a well-formed two-block record, with `overrides` applied."""
    qubit = level_boundary.ValueRef("entry", 0, "qubit")
    scalar = level_boundary.ValueRef("entry", 1, "scalar")
    branch = level_boundary.BlockRecord(
        "entry",
        (qubit, scalar),
        terminator="branch",
        edges=(level_boundary.Edge("join", (qubit, scalar)),),
    )
    join = level_boundary.BlockRecord("join", (qubit, scalar), terminator="return")
    record = level_boundary.ProgramRecord(
        functions=(level_boundary.FunctionRecord("main", (branch, join)),)
    )
    return dataclasses.replace(record, **overrides)  # type: ignore[arg-type]


def _codes(conversion: level_boundary.LevelConversion) -> set[str]:
    return {diagnostic.code for diagnostic in conversion.diagnostics}


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_static_circuit_round_trips_exactly():
    circuit = _static_circuit()
    conversion = level_boundary.round_trip(circuit)
    assert conversion.state == level_boundary.SUPPORTED_EXACT
    assert conversion.level == level_boundary.PUBLIC_LEVEL
    assert conversion.diagnostics == ()
    assert conversion.canonical_payload == circuit.to_dict()


def test_the_round_trip_carries_every_payload_section():
    payload = level_boundary.round_trip(_static_circuit()).canonical_payload
    assert payload is not None
    for section in (
        "instructions",
        "observables",
        "measurements",
        "metadata",
        "shape",
        "dtype",
    ):
        assert section in payload, section


def test_a_restored_circuit_is_equal_to_the_one_that_entered():
    circuit = _static_circuit()
    program = level_boundary.circuit_to_program(circuit)
    restored = level_boundary.program_to_circuit(program.record)
    assert restored.state == level_boundary.SUPPORTED_EXACT
    assert CircuitIR.from_dict(restored.canonical_payload) == circuit


def test_dynamic_circuit_lowers_and_round_trips_exactly():
    circuit = _dynamic_circuit()
    conversion = level_boundary.circuit_to_program(circuit)

    assert conversion.state == level_boundary.SUPPORTED_EXACT
    assert conversion.record is not None
    operations = conversion.record.functions[0].blocks[0].operations
    assert tuple(operation.opcode for operation in operations) == (
        "h",
        "measure",
        "reset",
        "measure",
        "x",
    )
    measurements = tuple(
        operation for operation in operations if operation.opcode == "measure"
    )
    assert tuple(result.kind for result in measurements[0].results) == (
        "qubit",
        "bool",
    )
    conditioned = operations[-1]
    assert {
        condition.value
        for clause in conditioned.condition_clauses
        for condition in clause
    } == {measurements[0].results[1], measurements[1].results[1]}
    restored = level_boundary.program_to_circuit(conversion.record)
    assert restored.state == level_boundary.SUPPORTED_EXACT
    assert restored.canonical_payload == circuit.to_dict()


def test_remeasuring_a_classical_slot_rebinds_subsequent_conditions():
    circuit = DynamicCircuit(3)
    circuit.measure(0, classical_bit=0)
    circuit.measure(1, classical_bit=0)
    circuit.conditional("x", 2, classical_bit=0)
    conversion = level_boundary.circuit_to_program(circuit.to_ir())

    assert conversion.record is not None
    operations = conversion.record.functions[0].blocks[0].operations
    measurements = tuple(
        operation for operation in operations if operation.opcode == "measure"
    )
    condition = operations[-1].condition_clauses[0][0]
    assert condition.value == measurements[1].results[1]
    assert condition.value != measurements[0].results[1]


def test_a_live_but_incorrect_dynamic_condition_fails_exact_round_trip():
    conversion = level_boundary.circuit_to_program(_dynamic_circuit())
    assert conversion.record is not None
    record = conversion.record
    entry = record.functions[0].blocks[0]
    conditioned = entry.operations[-1]
    first_clause = conditioned.condition_clauses[0]
    changed_clause = (
        dataclasses.replace(first_clause[0], equals=not first_clause[0].equals),
    )
    changed = dataclasses.replace(
        conditioned,
        condition_clauses=(changed_clause, *conditioned.condition_clauses[1:]),
    )
    edited_entry = dataclasses.replace(
        entry, operations=(*entry.operations[:-1], changed)
    )
    edited = dataclasses.replace(
        record,
        functions=(dataclasses.replace(record.functions[0], blocks=(edited_entry,)),),
    )

    assert level_boundary.verify_program_record(edited) == ()
    restored = level_boundary.program_to_circuit(edited)
    assert restored.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(restored) == {"level.round_trip_mismatch"}


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


def test_unsupported_dynamic_marker_is_refused():
    circuit = CircuitIR(
        n_wires=1,
        instructions=(Instruction("h", (0,), metadata={"is_dynamic": True}),),
    )
    conversion = level_boundary.classify_public_circuit(circuit)
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(conversion) == {"level.dynamic_instruction"}
    assert conversion.canonical_payload is None


def test_malformed_conditional_instruction_is_refused():
    circuit = CircuitIR(
        n_wires=2,
        instructions=(Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),),
    )
    conversion = level_boundary.classify_public_circuit(circuit)
    assert conversion.state == level_boundary.INVALID_INPUT
    assert _codes(conversion) == {"level.conditional_instruction"}


def test_a_non_circuit_input_is_invalid():
    conversion = level_boundary.classify_public_circuit(("h", 0))
    assert conversion.state == level_boundary.INVALID_INPUT
    assert _codes(conversion) == {"level.not_circuit_ir"}


def test_tampered_payload_is_refused():
    program = level_boundary.circuit_to_program(_static_circuit())
    payload = dict(program.record.canonical_payload or {})
    payload["instructions"] = "not a sequence of instructions"
    tampered = dataclasses.replace(program.record, canonical_payload=payload)
    conversion = level_boundary.program_to_circuit(tampered)
    assert conversion.state == level_boundary.INVALID_INPUT
    assert _codes(conversion) == {"level.payload_invalid"}


def test_a_record_without_a_recorded_payload_is_refused():
    conversion = level_boundary.program_to_circuit(_record_with())
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(conversion) == {"level.no_canonical_payload"}


def test_a_payload_that_is_not_canonical_is_refused():
    """A payload that decodes but re-encodes differently is not an exact round trip."""

    program = level_boundary.circuit_to_program(_static_circuit())
    payload = dict(program.record.canonical_payload or {})
    del payload["shape"]
    kept = level_boundary.ProgramRecord(
        functions=program.record.functions, canonical_payload=payload
    )
    conversion = level_boundary.program_to_circuit(kept)
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(conversion) == {"level.round_trip_mismatch"}
    assert level_boundary.round_trip(_static_circuit()).state == (
        level_boundary.SUPPORTED_EXACT
    )


def test_an_illegal_control_flow_graph_is_invalid_input():
    cases = {
        "program.entry_missing": _record_with(
            functions=(
                level_boundary.FunctionRecord(
                    "main", _record_with().functions[0].blocks, entry="nowhere"
                ),
            )
        ),
        "program.terminator_unknown": _record_with(
            functions=(
                level_boundary.FunctionRecord(
                    "main",
                    (
                        level_boundary.BlockRecord("entry", terminator="fallthrough"),
                        level_boundary.BlockRecord("join"),
                    ),
                ),
            )
        ),
        "program.edge_unknown": _record_with(
            functions=(
                level_boundary.FunctionRecord(
                    "main",
                    (
                        level_boundary.BlockRecord(
                            "entry",
                            terminator="branch",
                            edges=(level_boundary.Edge("absent"),),
                        ),
                    ),
                ),
            )
        ),
        "program.edge_unexpected": _record_with(
            functions=(
                level_boundary.FunctionRecord(
                    "main",
                    (
                        level_boundary.BlockRecord(
                            "entry",
                            terminator="return",
                            edges=(level_boundary.Edge("entry"),),
                        ),
                    ),
                ),
            )
        ),
        "program.edge_missing": _record_with(
            functions=(
                level_boundary.FunctionRecord(
                    "main", (level_boundary.BlockRecord("entry", terminator="branch"),)
                ),
            )
        ),
    }
    for code, record in cases.items():
        conversion = level_boundary.program_to_circuit(record)
        assert conversion.state == level_boundary.INVALID_INPUT, code
        assert code in _codes(conversion), (code, _codes(conversion))


def test_a_well_formed_record_is_accepted():
    assert level_boundary.verify_program_record(_record_with()) == ()


def test_a_graph_that_does_not_re_derive_its_payload_is_refused():
    """A payload is an attestation, not a source.

    Dropping the last operation from the graph leaves the recorded payload intact,
    so nothing about the payload itself is wrong. The restore has to notice that
    the graph no longer states the instruction sequence the payload records, which
    is the difference between re-deriving the public IR and trusting a copy of it.
    """

    program = level_boundary.circuit_to_program(_static_circuit())
    record = program.record
    entry = record.functions[0].blocks[0]
    edited = dataclasses.replace(
        record,
        functions=(
            dataclasses.replace(
                record.functions[0],
                blocks=(dataclasses.replace(entry, operations=entry.operations[:-1]),),
            ),
        ),
    )
    conversion = level_boundary.program_to_circuit(edited)
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(conversion) == {"level.round_trip_mismatch"}


def test_an_operation_that_does_not_type_is_invalid_input():
    """Every operation-level refusal names the code the contract declares for it."""

    qubit = level_boundary.ValueRef("entry", 0, "qubit")
    scalar = level_boundary.ValueRef("entry", 1, "scalar")
    cases = {
        "program.opcode_unknown": (
            level_boundary.Operation(
                "notagate", (qubit,), (level_boundary.ValueRef("entry", 2, "qubit"),)
            ),
        ),
        "program.operand_arity": (
            level_boundary.Operation(
                "cx",
                (qubit,),
                (
                    level_boundary.ValueRef("entry", 2, "qubit"),
                    level_boundary.ValueRef("entry", 3, "qubit"),
                ),
            ),
        ),
        "program.operand_kind": (
            level_boundary.Operation(
                "h", (scalar,), (level_boundary.ValueRef("entry", 2, "qubit"),)
            ),
        ),
        "program.parameter_missing": (
            level_boundary.Operation(
                "rz", (qubit,), (level_boundary.ValueRef("entry", 2, "qubit"),)
            ),
        ),
        "program.result_kind": (
            level_boundary.Operation(
                "h", (qubit,), (level_boundary.ValueRef("entry", 2, "scalar"),)
            ),
        ),
        "program.value_undefined": (
            level_boundary.Operation(
                "h",
                (level_boundary.ValueRef("elsewhere", 7, "qubit"),),
                (level_boundary.ValueRef("entry", 2, "qubit"),),
            ),
        ),
    }
    for code, operations in cases.items():
        block = level_boundary.BlockRecord(
            "entry", (qubit, scalar), terminator="return", operations=operations
        )
        record = level_boundary.ProgramRecord(
            functions=(level_boundary.FunctionRecord("main", (block,)),)
        )
        conversion = level_boundary.program_to_circuit(record)
        assert conversion.state == level_boundary.INVALID_INPUT, code
        assert code in _codes(conversion), (code, _codes(conversion))

    # A value is defined once per block, so a second definition of a live value is
    # refused even when the operation that defines it is otherwise well typed.
    first = level_boundary.Operation(
        "h", (qubit,), (level_boundary.ValueRef("entry", 2, "qubit"),)
    )
    second = level_boundary.Operation(
        "h", (qubit,), (level_boundary.ValueRef("entry", 2, "qubit"),)
    )
    repeated = level_boundary.BlockRecord(
        "entry",
        (qubit,),
        terminator="return",
        operations=(first, second),
    )
    conversion = level_boundary.program_to_circuit(
        level_boundary.ProgramRecord(
            functions=(level_boundary.FunctionRecord("main", (repeated,)),)
        )
    )
    assert "program.value_redefined" in _codes(conversion), _codes(conversion)

    # A compile constant that states no value cannot be reproduced as an operand.
    silent = level_boundary.BlockRecord(
        "entry",
        (scalar,),
        terminator="return",
        operations=(
            level_boundary.Operation(
                "constant",
                (),
                (level_boundary.ValueRef("entry", 5, "scalar"),),
                (),
                None,
            ),
        ),
    )
    quiet = level_boundary.ProgramRecord(
        functions=(level_boundary.FunctionRecord("main", (silent,)),)
    )
    assert "program.result_kind" in {
        diagnostic.code for diagnostic in level_boundary.verify_program_record(quiet)
    }


def test_the_contract_fake_types_dynamic_def_use():
    q0 = level_boundary.ValueRef("entry", 0, "qubit")
    q1 = level_boundary.ValueRef("entry", 1, "qubit")
    measured = level_boundary.ValueRef("entry", 2, "bool")
    q2 = level_boundary.ValueRef("entry", 3, "qubit")
    q3 = level_boundary.ValueRef("entry", 4, "qubit")
    block = level_boundary.BlockRecord(
        "entry",
        (q0,),
        operations=(
            level_boundary.Operation("measure", (q0,), (q1, measured)),
            level_boundary.Operation("reset", (q1,), (q2,)),
            level_boundary.Operation(
                "x",
                (q2,),
                (q3,),
                condition_clauses=((level_boundary.ConditionRef(measured, True),),),
            ),
        ),
    )
    record = level_boundary.ProgramRecord(
        functions=(level_boundary.FunctionRecord("main", (block,)),)
    )

    assert level_boundary.verify_program_record(record) == ()
    measure = level_boundary.opcode_signature("measure")
    reset = level_boundary.opcode_signature("reset")
    assert measure is not None
    assert measure.operand_kinds == ("qubit",)
    assert measure.result_kinds == ("qubit", "bool")
    assert reset is not None
    assert reset.operand_kinds == ("qubit",)
    assert reset.result_kinds == ("qubit",)


def test_dynamic_condition_contract_rejects_invalid_def_use():
    q0 = level_boundary.ValueRef("entry", 0, "qubit")
    q1 = level_boundary.ValueRef("entry", 1, "qubit")
    boolean = level_boundary.ValueRef("entry", 2, "bool")
    scalar = level_boundary.ValueRef("entry", 3, "scalar")
    absent = level_boundary.ValueRef("entry", 99, "bool")

    cases = {
        "program.condition_kind": level_boundary.Operation(
            "x",
            (q0,),
            (q1,),
            condition_clauses=((level_boundary.ConditionRef(scalar, True),),),
        ),
        "program.condition_undefined": level_boundary.Operation(
            "x",
            (q0,),
            (q1,),
            condition_clauses=((level_boundary.ConditionRef(absent, True),),),
        ),
        "program.condition_target": level_boundary.Operation(
            "reset",
            (q0,),
            (q1,),
            condition_clauses=((level_boundary.ConditionRef(boolean, True),),),
        ),
        "program.condition_noncanonical": level_boundary.Operation(
            "x", (q0,), (q1,), condition_clauses=((),)
        ),
        "program.condition_value": level_boundary.Operation(
            "x",
            (q0,),
            (q1,),
            condition_clauses=(
                (level_boundary.ConditionRef(boolean, 1),),  # type: ignore[arg-type]
            ),
        ),
    }
    for expected, operation in cases.items():
        block = level_boundary.BlockRecord(
            "entry", (q0, boolean, scalar), operations=(operation,)
        )
        record = level_boundary.ProgramRecord(
            functions=(level_boundary.FunctionRecord("main", (block,)),)
        )
        codes = {
            diagnostic.code
            for diagnostic in level_boundary.verify_program_record(record)
        }
        assert expected in codes, (expected, codes)


# --------------------------------------------------------------------------
# Value identity, linearity, and joins
# --------------------------------------------------------------------------


def test_value_identity_is_scope_and_index():
    unnamed = level_boundary.BlockRecord(
        "entry", (level_boundary.ValueRef("", 0),), terminator="return"
    )
    negative = level_boundary.BlockRecord(
        "entry", (level_boundary.ValueRef("entry", -1),), terminator="return"
    )
    unknown_kind = level_boundary.BlockRecord(
        "entry", (level_boundary.ValueRef("entry", 0, "qudit"),), terminator="return"
    )
    for block in (unnamed, negative, unknown_kind):
        record = level_boundary.ProgramRecord(
            functions=(level_boundary.FunctionRecord("main", (block,)),)
        )
        assert "value.identity" in {
            diagnostic.code
            for diagnostic in level_boundary.verify_program_record(record)
        }


def test_linear_value_has_exactly_one_consumer():
    qubit = level_boundary.ValueRef("entry", 0, "qubit")
    duplicated = level_boundary.BlockRecord(
        "entry",
        (qubit,),
        terminator="branch",
        edges=(
            level_boundary.Edge("join", (qubit,)),
            level_boundary.Edge("join", (qubit,)),
        ),
    )
    record = level_boundary.ProgramRecord(
        functions=(
            level_boundary.FunctionRecord(
                "main",
                (duplicated, level_boundary.BlockRecord("join", (qubit,))),
            ),
        )
    )
    assert "value.linear_duplicate" in {
        diagnostic.code for diagnostic in level_boundary.verify_program_record(record)
    }


def test_linear_value_must_be_forwarded_by_a_branch():
    qubit = level_boundary.ValueRef("entry", 0, "qubit")
    dropped = level_boundary.BlockRecord(
        "entry",
        (qubit,),
        terminator="branch",
        edges=(level_boundary.Edge("join"),),
    )
    record = level_boundary.ProgramRecord(
        functions=(
            level_boundary.FunctionRecord(
                "main", (dropped, level_boundary.BlockRecord("join"))
            ),
        )
    )
    assert "value.linear_unconsumed" in {
        diagnostic.code for diagnostic in level_boundary.verify_program_record(record)
    }


def test_a_returning_block_consumes_its_linear_argument():
    assert level_boundary.verify_program_record(_record_with()) == ()


def test_non_linear_values_may_have_many_consumers():
    scalar = level_boundary.ValueRef("entry", 0, "scalar")
    block = level_boundary.BlockRecord(
        "entry",
        (scalar, scalar),
        terminator="branch",
        edges=(level_boundary.Edge("join", (scalar, scalar)),),
    )
    record = level_boundary.ProgramRecord(
        functions=(
            level_boundary.FunctionRecord(
                "main", (block, level_boundary.BlockRecord("join", (scalar, scalar)))
            ),
        )
    )
    assert level_boundary.verify_program_record(record) == ()


def test_join_requires_matching_argument_kinds():
    qubit = level_boundary.ValueRef("entry", 0, "qubit")
    scalar = level_boundary.ValueRef("entry", 1, "scalar")
    boolean = level_boundary.ValueRef("entry", 1, "bool")
    block = level_boundary.BlockRecord(
        "entry",
        (qubit, scalar),
        terminator="branch",
        edges=(level_boundary.Edge("join", (qubit, scalar)),),
    )
    arity = level_boundary.ProgramRecord(
        functions=(
            level_boundary.FunctionRecord(
                "main", (block, level_boundary.BlockRecord("join", (qubit,)))
            ),
        )
    )
    kinds = level_boundary.ProgramRecord(
        functions=(
            level_boundary.FunctionRecord(
                "main", (block, level_boundary.BlockRecord("join", (qubit, boolean)))
            ),
        )
    )
    assert "program.edge_arity" in {
        diagnostic.code for diagnostic in level_boundary.verify_program_record(arity)
    }
    assert "program.edge_kind" in {
        diagnostic.code for diagnostic in level_boundary.verify_program_record(kinds)
    }


# --------------------------------------------------------------------------
# Support states
# --------------------------------------------------------------------------


def test_every_refusal_names_a_declared_code():
    refusals = [
        level_boundary.classify_public_circuit(None),
        level_boundary.classify_public_circuit(
            CircuitIR(
                n_wires=1,
                instructions=(Instruction("h", (0,), metadata={"is_dynamic": True}),),
            )
        ),
        level_boundary.program_to_circuit("not a record"),
        level_boundary.program_to_circuit(_record_with()),
        level_boundary.lower_to_level(_static_circuit(), level_boundary.LEVEL_QUANTUM),
        level_boundary.lower_to_level(_static_circuit(), level_boundary.LEVEL_TARGET),
    ]
    refused = 0
    for conversion in refusals:
        assert conversion.state != level_boundary.SUPPORTED_EXACT
        assert conversion.diagnostics
        for diagnostic in conversion.diagnostics:
            assert diagnostic.code in level_boundary.DIAGNOSTIC_LEVELS, diagnostic.code
            assert diagnostic.level == level_boundary.DIAGNOSTIC_LEVELS[diagnostic.code]
            refused += 1
    assert refused >= len(refusals)


def test_a_supported_result_carries_no_diagnostic_and_a_refusal_carries_one():
    with pytest.raises(level_boundary.LevelContractError):
        level_boundary.LevelConversion(
            state=level_boundary.SUPPORTED_EXACT,
            level=level_boundary.PUBLIC_LEVEL,
            diagnostics=(level_boundary.LevelDiagnostic("x", "y", "circuit"),),
        )
    with pytest.raises(level_boundary.LevelContractError):
        level_boundary.LevelConversion(
            state=level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS,
            level=level_boundary.PUBLIC_LEVEL,
        )


def test_no_best_effort_state_exists():
    assert "best_effort" not in level_boundary.SUPPORT_STATES
    with pytest.raises(level_boundary.LevelContractError):
        level_boundary.LevelConversion(
            state="best_effort", level=level_boundary.PUBLIC_LEVEL
        )


def test_lower_to_level_realizes_only_declared_exits():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert tuple(contract["realized_exits"]) == level_boundary.REALIZED_EXITS
    assert tuple(contract["declared_but_unimplemented_exits"]) == tuple(
        level
        for level in level_boundary.INTERNAL_LEVELS
        if level not in level_boundary.REALIZED_EXITS
    )
    for level in contract["declared_but_unimplemented_exits"]:
        conversion = level_boundary.lower_to_level(_static_circuit(), level)
        assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
        assert _codes(conversion) == {"level.unsupported_exit"}
    with pytest.raises(level_boundary.LevelContractError):
        level_boundary.lower_to_level(_static_circuit(), "gpu")


# --------------------------------------------------------------------------
# This boundary is internal
# --------------------------------------------------------------------------


def test_the_boundary_is_not_a_root_export():
    names = (
        "ProgramRecord",
        "LevelConversion",
        "lower_to_level",
        "ValueRef",
        "DIAGNOSTIC_LEVELS",
    )
    for name in names:
        assert name not in fq.__all__, name
        assert name not in level_boundary.__dict__.get("__all__", ()), name
        assert not hasattr(level_boundary.CircuitIR, name), name
    snapshot = PUBLIC_SNAPSHOT.read_text(encoding="utf-8")
    for name in names:
        assert name not in snapshot, name


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_boundary_gate_accepts_the_checked_in_contract():
    assert _GATE.contract_errors(_contract()) == ()


def test_the_boundary_gate_runs_in_ci_and_before_push():
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_multi_level_ir_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_multi_level_ir_contract.py"' in pre_push
