"""Conformance suite for the internal multi-level IR boundary.

`contracts/multi-level-ir-internal-v1-candidate.json` declares seven claims about
the boundary between the public IR and the internal program, quantum, and target
levels: level entry and exit, value identity, linearity, joins, failure
categories, round trip, and rejection. `flagquantum.core.ir.levels` supplies the
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


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


def test_dynamic_instruction_is_refused():
    circuit = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction("h", (0,)),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
        ),
    )
    conversion = level_boundary.classify_public_circuit(circuit)
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
    assert _codes(conversion) == {"level.dynamic_instruction"}
    assert conversion.canonical_payload is None


def test_conditional_instruction_is_refused():
    circuit = CircuitIR(
        n_wires=2,
        instructions=(Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),),
    )
    conversion = level_boundary.classify_public_circuit(circuit)
    assert conversion.state == level_boundary.UNSUPPORTED_WITH_DIAGNOSTICS
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
