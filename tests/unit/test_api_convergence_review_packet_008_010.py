from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "contracts" / "api-convergence-review-packet-008-010.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_packet_records_exact_api_owner_approval() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))

    assert packet["status"] == "approved"
    assert packet["approval_record"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-09-01",
        "approval_directive": "approve 008-010",
        "approved_proposals": ["008", "009", "010"],
        "held_proposals": [],
        "overall_freeze_approved": False,
    }
    assert packet["rules"]["continue_or_do_is_not_freeze_approval"] is True
    assert [item["proposal"] for item in packet["proposals"]] == ["008", "009", "010"]

    for item in packet["proposals"]:
        assert _sha256(ROOT / item["contract"]) == item["contract_sha256"]
        assert _sha256(ROOT / item["proposal_document"]) == item["proposal_sha256"]
        assert item["current_decision"] == "approved"
        assert item["reviewed_contract_sha256"] != item["contract_sha256"]
        assert item["reviewed_proposal_sha256"] != item["proposal_sha256"]

    contracts = [
        json.loads((ROOT / item["contract"]).read_text(encoding="utf-8"))
        for item in packet["proposals"]
    ]
    assert contracts[0]["status"] == "governance_baseline_frozen"
    assert contracts[0]["rules"]["candidate_is_frozen_contract"] is False
    assert contracts[0]["rules"]["governance_baseline_is_frozen"] is True
    assert [contract["status"] for contract in contracts[1:]] == ["frozen", "frozen"]
    assert all(
        contract["rules"]["candidate_is_frozen_contract"] is True
        for contract in contracts[1:]
    )


def test_review_packet_preserves_prior_signed_assets_exactly() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))

    for key in ("protected_prior_approval", "protected_public_api_candidate"):
        protected = packet[key]
        assert protected["modified_by_this_review"] is False
        assert _sha256(ROOT / protected["path"]) == protected["sha256"]


def test_governance_approval_does_not_freeze_experimental_features() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    proposal = packet["proposals"][0]

    assert proposal["approval_kind"] == "governance_baseline"
    assert "nested experimental feature symbols" in proposal["not_frozen_on_approval"]
    assert "experimental_feature_compatibility" in packet["explicit_non_approvals"]
    contract = json.loads((ROOT / proposal["contract"]).read_text(encoding="utf-8"))
    assert contract["compatibility_guarantee"] is False
    assert contract["rules"]["nested_experimental_features_are_frozen"] is False
