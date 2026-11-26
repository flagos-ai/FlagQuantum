"""Approved expression extensions retain prior bytes, hashes and binding semantics."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter, ParameterExpression

try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:
    tomllib = importlib.import_module("tomli")

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/parameter_expression_compatibility.json"
PROPOSAL128_FIXTURE = (
    ROOT / "tests/fixtures/parameter_expression_proposal128_compatibility.json"
)
FROZEN128 = json.loads(PROPOSAL128_FIXTURE.read_text())


@pytest.mark.parametrize(
    "case",
    [*json.loads(FIXTURE.read_text()), *FROZEN128["cases"]],
    ids=lambda c: c["name"],
)
def test_legacy_expression_encoding_hash_and_binding(case: dict[str, object]) -> None:
    encoded = str(case["json"])
    circuit = CircuitIR.from_json(encoded)
    assert circuit.to_dict() == case["payload"]
    assert circuit.to_json() == encoded
    assert circuit.content_hash == case["content_hash"]
    value = circuit.instructions[0].params["theta"]
    assert value.bind(case["binding"]) == case["value"]


@pytest.mark.parametrize("case", FROZEN128["failures"], ids=lambda c: c["name"])
def test_proposal128_preserves_old_failure_categories_and_diagnostics(
    case: dict[str, Any],
) -> None:
    args = tuple(Parameter("theta") if arg == "theta" else arg for arg in case["args"])
    value = case["binding"]
    if isinstance(value, dict):
        value = complex(*value["complex"])
    assert case["error_type"] == "ValueError"
    with pytest.raises(ValueError) as raised:
        ParameterExpression(case["op"], args).bind({"theta": value})
    assert str(raised.value) == case["message"]


def test_proposal128_keeps_prior_fixture_and_parameter_lookup_precedence() -> None:
    assert (
        hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        == FROZEN128["legacy_fixture_sha256"]
    )
    theta = Parameter("theta")
    bindings: dict[str | Parameter, float] = {"theta": 0.375, theta: 0.75}
    assert (theta + 2).bind(bindings) == FROZEN128["parameter_key_precedence"]


def test_approved_contract_names_expression_owners_and_arity() -> None:
    contract = tomllib.loads(
        (ROOT / "contracts/parameter-expression-contract.toml").read_text()
    )
    assert contract["version"] == "1.1"
    assert contract["approval"] == "explicit_user_approval_2026-11-26"
    assert contract["public_methods_added"] == []
    assert contract["expression"]["unary"] == [
        "neg",
        "sin",
        "cos",
        "exp",
        "log",
        "acos",
        "floor",
    ]
    assert contract["expression"]["binary"] == ["add", "sub", "mul", "pow", "div"]
    assert contract["serialization"]["fields"] == ["op", "args"]
    assert contract["adapters"]["foreign_objects_in_core"] is False
