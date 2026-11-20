"""Proposal 110 preserves existing integer feedback artifacts byte for byte."""

import json
from pathlib import Path

import pytest

from flagquantum.compiler.registers.component_feedback import (
    classical_bit_feedback_function,
)
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_signed_register_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def existing_artifacts() -> dict[str, object]:
    result: dict[str, object] = {}
    for kind, dtype, value in (
        ("unsigned", classical_unsigned_register_type(3), 5),
        ("signed", classical_signed_register_type(3), -2),
    ):
        function = classical_bit_feedback_function(
            "main",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("h", (0,)),)),
            (),
        )
        module = classical_control_component_program(
            ProgramRecord((function,)), (((), (value,)),), control_types=(dtype,)
        )
        bound = bind_register_program(
            module, ((), (0,)), classical_inputs=(value, None)
        )
        result[kind] = {
            "program_repr": repr(module),
            "circuit_json": bound.circuit.to_json(),
            "content_hash": bound.circuit.content_hash,
            "outputs": [list(word) for word in bound.outputs],
            "classical_outputs": [list(word) for word in bound.classical_outputs],
        }
    return result


def test_previous_word_artifacts_are_identical() -> None:
    fixture = json.loads(
        (
            ROOT / "tests/fixtures/classical_field_feedback_compatibility.json"
        ).read_text()
    )
    assert existing_artifacts() == fixture


def test_explicitly_approved_classical_field_contract() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/classical-field-feedback-contract.toml").read_text()
    )
    assert contract["approval"] == "explicit_user_approval_2026-11-20"
    assert contract["public_methods_added"] == []
    assert contract["read"]["new_classical_slots"] == 0
    assert contract["read"]["opcode"] == "read_classical_register"
    assert contract["compatibility"]["operation_fields"] == "unchanged"
