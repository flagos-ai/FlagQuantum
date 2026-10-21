"""Fixed pre-081 Program IR identity and existing plan compatibility evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir.program.model import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
)

pytestmark = pytest.mark.unit


def test_old_record_fields_and_calling_conventions_remain_prefixes() -> None:
    path = Path(__file__).parents[1] / "fixtures/rydberg_program_legacy.json"
    evidence = json.loads(path.read_text())
    for entry in evidence["captures"].values():
        capture = entry["capture"]
        for record, key, approved_tail in (
            (Operation, "operation_fields", ["physical_data"]),
            (ProgramRecord, "program_fields", ["physical_context"]),
        ):
            names = [item.name for item in fields(record)]
            old = capture[key]
            assert names[: len(old)] == old
            assert names[len(old) :] in ([], approved_tail)
            signature = inspect.signature(record)
            assert list(signature.parameters) == names
            signature_key = (
                "operation_signature" if record is Operation else "program_signature"
            )
            old_prefix = capture[signature_key].removesuffix(") -> None")
            assert str(signature).startswith(old_prefix)
            for name in names[len(old) :]:
                assert signature.parameters[name].default is None
        assert capture["canonical_payload_excluded_from_equality"]
    legacy = (
        Path(__file__).parents[1] / "fixtures/lindblad_output_plan_compatibility.json"
    )
    assert (
        hashlib.sha256(legacy.read_bytes()).hexdigest()
        == evidence["legacy_plan_fixture_sha256"]
    )


@pytest.mark.parametrize("opcode", ["h", "constant", "rydberg.move", "rydberg.pulse"])
def test_ordinary_record_hashes_keep_the_old_exact_tuple_formula(opcode: str) -> None:
    incoming = ValueRef("entry", 0, "qubit")
    outgoing = ValueRef("entry", 1, "qubit")
    op = Operation(opcode, (incoming,), (outgoing,))
    assert hash(op) == hash(
        (
            op.opcode,
            op.operands,
            op.results,
            op.parameters,
            op.literal,
            op.condition_clauses,
        )
    )
    functions = (FunctionRecord("main", (BlockRecord("entry", operations=(op,)),)),)
    original = ProgramRecord(functions)
    different_payload = ProgramRecord(functions, {"unrelated": "payload"})
    assert original == different_payload
    assert hash(original) == hash(different_payload) == hash((functions,))


@pytest.mark.parametrize("name", ["rydberg.move", "rydberg.pulse"])
def test_physical_names_do_not_capture_existing_custom_matrix_gates(name: str) -> None:
    circuit = fq.Circuit(1)
    circuit.any(0, matrix=torch.eye(2, dtype=torch.complex64), name=name)
    result = fq.run(circuit, outputs=fq.probabilities())
    torch.testing.assert_close(
        result.probabilities, torch.tensor([[1.0, 0.0]]), atol=0, rtol=0
    )
