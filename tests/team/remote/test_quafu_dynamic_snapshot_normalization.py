"""Offline conformance for Quafu dynamic capability normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

import flagquantum as fq
from flagquantum.core.target_capabilities import TargetCapabilitySnapshot
from flagquantum.remote.qpu.quafu_dynamic import (
    _quafu_dynamic_snapshot_from_fixture,
)
from flagquantum.runtime.dynamic.remote_capabilities import DYNAMIC_REMOTE_EXTENSION

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT / "tests" / "fixtures" / "quafu_dynamic_capability_complete_candidate.json"
)
QUAFU_EXTENSION = "org.flagquantum.quafu-dynamic-v1"


def _fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Quafu capability fixture must be a JSON object")
    return cast(dict[str, Any], payload)


def _remove(payload: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    parent: dict[str, Any] = payload
    for part in parts[:-1]:
        parent = cast(dict[str, Any], parent[part])
    del parent[parts[-1]]


def test_complete_fixture_produces_one_deterministic_exact_snapshot() -> None:
    first = _quafu_dynamic_snapshot_from_fixture(_fixture())
    second = _quafu_dynamic_snapshot_from_fixture(_fixture())

    assert first == second
    assert first == TargetCapabilitySnapshot.from_json(first.to_json())
    assert first.target_identity.to_dict() == {
        "target_id": "quafu:Baihua",
        "target_class": "remote_qpu",
        "provider": "quafu",
        "provider_version": "quafu-contract-fixture-1",
        "target_revision": "fixture-calibration-1",
        "environment_id": "offline-contract-fixture",
    }
    assert first.captured_at == "2026-10-24T10:00:00Z"
    assert first.valid_until == "2026-10-24T11:00:00Z"
    assert len(first.snapshot_id) == 64


def test_fixture_projects_core_dynamic_and_quafu_owned_facts() -> None:
    snapshot = _quafu_dynamic_snapshot_from_fixture(_fixture())
    payload = snapshot.to_dict()
    facts = {fact["name"]: fact for fact in payload["facts"]}

    assert facts["qubits.logical_capacity"]["value"] == 4
    assert facts["qubits.physical_capacity"]["value"] == 4
    assert facts["gates.native"]["value"] == ["cz", "h", "rx", "ry", "rz"]
    dynamic = payload["extensions"][DYNAMIC_REMOTE_EXTENSION]["facts"]
    assert dynamic["mid_circuit_measurement"]["state"] == "supported"
    assert dynamic["classical_feed_forward"]["state"] == "supported"
    assert dynamic["supported_conditional_gates"]["value"] == ["rx", "rz"]
    assert payload["extensions"][QUAFU_EXTENSION] == {
        "schema_version": "1.0",
        "attestation_scope": "conformance_fixture_only",
        "program": {"dialect": "openqasm", "version": "3.0"},
        "directed_connectivity": [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
        ],
    }


@pytest.mark.parametrize(
    "path",
    (
        "backend.native_operations",
        "backend.directed_connectivity",
        "program.dialect",
        "program.version",
        "dynamic.mid_circuit_measurement",
        "dynamic.reset",
        "dynamic.repeated_measurement",
        "dynamic.classical_feed_forward",
        "dynamic.conditional_operations",
        "dynamic.condition_width",
        "dynamic.condition_values",
        "dynamic.condition_disjunction",
        "dynamic.mid_circuit_result_data",
    ),
)
def test_missing_authoritative_fact_fails_closed(path: str) -> None:
    payload = copy.deepcopy(_fixture())
    _remove(payload, path)

    with pytest.raises(
        ValueError,
        match=f"quafu_dynamic_capability_evidence_unavailable:{path}",
    ):
        _quafu_dynamic_snapshot_from_fixture(payload)


def test_non_fixture_attestation_cannot_be_normalized() -> None:
    payload = _fixture()
    payload["attestation_scope"] = "live_hardware"

    with pytest.raises(ValueError, match="conformance_fixture_only"):
        _quafu_dynamic_snapshot_from_fixture(payload)


def test_fixture_normalizer_is_not_a_public_api() -> None:
    assert "quafu_dynamic_snapshot_from_fixture" not in fq.__all__
