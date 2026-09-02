from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase1-batch-f-performance-remediation-authorization.json"
)
ATTESTATION = ROOT / "contracts/ir-phase1-batch-f-performance-remediation.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_performance_remediation_binds_authorization_and_original_blocker() -> None:
    authorization = _load(AUTHORIZATION)

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve IR-PHASE1-BATCH-F-PERFORMANCE-REMEDIATION"
    )
    assert (
        _sha256(ROOT / authorization["batch_f_authorization"]["path"])
        == authorization["batch_f_authorization"]["sha256"]
    )
    assert _sha256(ROOT / authorization["blocker"]["path"]) == (
        authorization["blocker"]["sha256"]
    )
    assert authorization["approved_change"]["budget_change_allowed"] is False


def test_successor_preserves_history_and_binds_current_importer() -> None:
    attestation = _load(ATTESTATION)
    historical = attestation["historical_candidate"]
    successor = attestation["successors"][
        "flagquantum/_compiler/importers/circuit_ir.py"
    ]

    assert _sha256(ROOT / attestation["authorization"]["path"]) == (
        attestation["authorization"]["sha256"]
    )
    assert _sha256(ROOT / historical["path"]) == historical["sha256"]
    assert historical["mutated"] is False
    assert successor["previous_sha256"] == (
        "3f8d18cccb05e5bf267d58aba3fbff3baa3579257c92da131f358fb71e606770"
    )
    assert _sha256(ROOT / "flagquantum/_compiler/importers/circuit_ir.py") == (
        successor["current_sha256"]
    )
    assert successor["semantic_output_change"] is False


def test_retest_passes_every_unchanged_budget() -> None:
    attestation = _load(ATTESTATION)
    budget = attestation["budget"]

    assert attestation["status"] == "passed"
    assert budget["changed"] is False
    assert _sha256(ROOT / budget["path"]) == budget["sha256"]
    assert attestation["retest"]["growth_passed"] is True
    assert all(
        case["latency_passed"] and case["memory_passed"]
        for case in attestation["retest"]["cases"]
    )
