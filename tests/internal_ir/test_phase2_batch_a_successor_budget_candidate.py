from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "contracts/ir-phase2-batch-a-successor-budget-review-candidate.json"
EVIDENCE = ROOT / "contracts/ir-phase2-batch-a-performance-remediation-evidence.json"
ORIGINAL = ROOT / "tests/fixtures/internal_ir/phase2_performance_budget_candidate.json"
SUCCESSOR = (
    ROOT
    / "tests/fixtures/internal_ir/phase2_performance_budget_successor_candidate.json"
)
ARTIFACT_SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-a-performance-remediation-artifact-successor.json"
)
BATCH_B_SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-b-authorized-artifact-successor.json"
)


def test_successor_budget_review_binds_every_remediation_artifact() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "awaiting_owner_approval"
    assert review["approval_command"] == ("approve IR-PHASE2-BATCH-A-SUCCESSOR-BUDGET")
    for relative_path, expected_hash in review["artifacts"].items():
        artifact = ROOT / relative_path
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            continue
        successor = json.loads(BATCH_B_SUCCESSOR.read_text(encoding="utf-8"))
        transition = successor["candidates"][str(REVIEW.relative_to(ROOT))][
            "artifact_transitions"
        ][relative_path]
        assert transition["predecessor_sha256"] == expected_hash
        assert transition["successor_sha256"] == actual_hash


def test_artifact_successor_is_authorized_and_keeps_historical_reviews_immutable() -> (
    None
):
    successor = json.loads(ARTIFACT_SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]

    assert hashlib.sha256((ROOT / authorization["path"]).read_bytes()).hexdigest() == (
        authorization["sha256"]
    )
    assert successor["predecessor_review_semantics_changed"] is False
    assert successor["public_or_default_path_changed"] is False
    for relative_path, candidate in successor["candidates"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            candidate["sha256"]
        )


def test_successor_candidate_is_immutable_and_preserves_original_snapshot() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert successor["status"] == "pending_owner_approval"
    assert successor["approval"]["approved"] is False
    assert successor["supersedes_if_approved"] == str(ORIGINAL.relative_to(ROOT))
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == (
        review["artifacts"][str(ORIGINAL.relative_to(ROOT))]
    )
    assert review["proposed_decisions"]["batch_a_exit_authorized"] is False
    assert review["proposed_decisions"]["batch_b_start_authorized"] is False


def test_successor_budget_has_measured_headroom_without_public_sla_claim() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    measured = {item["gate_count"]: item for item in evidence["full_pipeline"]}
    component = {item["gate_count"]: item for item in evidence["component_p95_ms"]}

    assert "not a public SLA" in successor["derivation"]["claim_boundary"]
    for budget in successor["budgets"]:
        gate_count = budget["gate_count"]
        observed_latency = max(
            measured[gate_count]["p95_ms"], component[gate_count]["total"]
        )
        assert budget["pipeline_p95_ms_max"] >= observed_latency * 1.25
        assert budget["peak_host_memory_bytes_max"] >= (
            measured[gate_count]["peak_host_memory_bytes"] * 1.25
        )
