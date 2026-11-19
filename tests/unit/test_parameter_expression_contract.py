"""Proposal 107 keeps previous expression bytes, hashes and binding semantics."""

import json
import tomllib
from pathlib import Path

import pytest

from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/parameter_expression_compatibility.json"


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text()), ids=lambda c: c["name"]
)
def test_legacy_expression_encoding_hash_and_binding(case: dict[str, object]) -> None:
    encoded = str(case["json"])
    circuit = CircuitIR.from_json(encoded)
    assert circuit.to_dict() == case["payload"]
    assert circuit.to_json() == encoded
    assert circuit.content_hash == case["content_hash"]
    value = circuit.instructions[0].params["theta"]
    assert value.bind(case["binding"]) == case["value"]


def test_approved_contract_names_expression_owners_and_arity() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/parameter-expression-contract.toml").read_text()
    )
    assert contract["approval"] == "explicit_user_approval_2026-11-19"
    assert contract["public_methods_added"] == []
    assert contract["expression"]["unary"] == ["neg", "sin", "cos", "exp", "log"]
    assert contract["expression"]["binary"] == ["add", "sub", "mul", "pow", "div"]
    assert contract["serialization"]["fields"] == ["op", "args"]
    assert contract["adapters"]["foreign_objects_in_core"] is False
