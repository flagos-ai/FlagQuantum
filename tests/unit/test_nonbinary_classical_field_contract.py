"""Proposal 111 retains accepted binary typed-feedback artifacts exactly."""

import json
from pathlib import Path

import pytest

from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.test_classical_field_feedback import fixture
from tests.unit.test_classical_field_feedback_contract import existing_artifacts
from tests.unit.test_component_classical_control import hierarchy

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


def binary_artifacts() -> dict[str, object]:
    artifacts = existing_artifacts()
    for width, degree in ((1, None), (3, None), (1, 0), (2, 1)):
        codec, _ = fixture(width, degree)
        dtype = codec.register_type(classical=True)
        value = (1 << int(codec.bitsize)) - 1
        child, _ = hierarchy()
        record = classical_control_component_program(
            child, (((), (value,)),), control_types=(dtype,)
        )
        bound = bind_register_program(
            record, ((), (0,), (1,)), classical_inputs=(value, None, None), clean=(2,)
        )
        artifacts[f"field_{width}_{degree}"] = {
            "program_repr": repr(record),
            "circuit_json": bound.circuit.to_json(),
            "content_hash": bound.circuit.content_hash,
            "outputs": [list(word) for word in bound.outputs],
            "classical_outputs": [list(word) for word in bound.classical_outputs],
        }
    return artifacts


def test_pre_extension_binary_artifacts_remain_exact() -> None:
    expected = json.loads(
        (
            ROOT / "tests/fixtures/nonbinary_classical_field_compatibility.json"
        ).read_text()
    )
    assert binary_artifacts() == expected


def test_nonbinary_classical_contract_is_explicitly_approved() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/nonbinary-classical-field-contract.toml").read_text()
    )
    assert contract["approval"] == "explicit_user_approval_2026-11-20"
    assert contract["public_methods_added"] == []
    assert contract["read"]["new_classical_slots"] == 0
    assert contract["admission"]["quantum_field_scope"] == "existing_binary_only"
    assert contract["reference"]["polynomial_tensor_index_discrepancy"] == "retained"
