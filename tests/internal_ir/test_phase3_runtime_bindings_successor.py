from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler import runtime_abi
from flagquantum._compiler.runtime_abi import RuntimeBindings, RuntimeParameterBinding

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase3-runtime-bindings-successor-authorization.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization() -> dict[str, object]:
    return json.loads(AUTHORIZATION.read_text(encoding="utf-8"))


def test_successor_binds_immutable_predecessors_and_exact_artifacts() -> None:
    authorization = _authorization()

    assert authorization["status"] == "authorized_artifact_successor"
    assert authorization["historical_records_mutated"] is False
    for relative_path, expected_hash in authorization["predecessor_candidates"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
    for relative_path, transition in authorization["artifact_transitions"].items():
        assert _sha256(ROOT / relative_path) == transition["successor_sha256"]
        assert transition["predecessor_sha256"] != transition["successor_sha256"]


def test_private_runtime_bindings_are_distinct_from_public_options() -> None:
    bindings = RuntimeBindings(
        shots=128,
        parameter_bindings=(RuntimeParameterBinding("theta", 0.5),),
    )

    assert bindings.to_dict() == {
        "shots": 128,
        "parameter_bindings": [{"name": "theta", "value": 0.5}],
    }
    assert fq.ExecutionOptions is not RuntimeBindings
    assert not hasattr(runtime_abi, "ExecutionOptions")


def test_successor_preserves_wire_and_identity_semantics() -> None:
    preserved = _authorization()["preserved_contracts"]

    assert preserved == {
        "public_execution_options_unchanged": True,
        "serialized_options_key_unchanged": True,
        "binding_identity_inputs_unchanged": True,
        "runtime_behavior_unchanged": True,
        "public_exports_changed": False,
        "default_execution_path_changed": False,
    }
