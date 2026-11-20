"""Proposal 108 preserves the exact pre-extension CBit graph and bound stream."""

import json
from pathlib import Path

import pytest

from flagquantum.compiler.registers.component_feedback import (
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_existing_cbit_graph_bytes_hash_and_slots() -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/classical_word_cbit_compatibility.json").read_text()
    )
    function = classical_bit_feedback_function(
        "main",
        (Register("q", quantum_bit_type()),),
        CircuitIR(1, (Instruction("h", (0,)),)),
        (((), (True,)),),
    )
    bound = bind_register_program(
        ProgramRecord((function,)), ((), (0,)), classical_inputs=(1, None)
    )
    assert repr(function) == fixture["function_repr"]
    assert bound.circuit.to_json() == fixture["circuit_json"]
    assert bound.circuit.content_hash == fixture["content_hash"]
    assert [list(word) for word in bound.outputs] == fixture["outputs"]
    assert [list(word) for word in bound.classical_outputs] == fixture[
        "classical_outputs"
    ]


def test_approved_contract_preserves_read_opcode_and_owners() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/classical-word-feedback-contract.toml").read_text()
    )
    assert contract["approval"] == "explicit_user_approval_2026-11-20"
    assert contract["public_methods_added"] == []
    assert contract["read"]["opcode"] == "read_classical_register"
    assert contract["read"]["new_classical_slots"] == 0
    assert (
        contract["read"]["results"]
        == "identical_type_successor_then_width_boolean_snapshots"
    )
    assert contract["compatibility"]["operation_fields"] == "unchanged"
