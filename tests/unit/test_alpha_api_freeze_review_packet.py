from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "contracts" / "alpha-api-freeze-review-packet.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_packet_binds_exact_approved_contracts_without_overall_freeze() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))

    assert packet["status"] == "contract_freeze_approved"
    assert packet["approval_record"] == {
        "approved_by": "API owner via explicit user directive",
        "approved_at": "2026-09-01",
        "approved_proposals": ["003", "004", "005", "006", "007"],
        "held_proposals": [],
        "overall_freeze_approved": False,
    }
    assert packet["rules"]["silence_is_not_approval"] is True

    manifest = packet["stable_core"]
    assert _sha256(ROOT / manifest["manifest"]) == manifest["sha256"]
    candidate = packet["public_api_candidate"]
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]

    contracts = packet["contracts"]
    assert [item["proposal"] for item in contracts] == [
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
    ]
    for item in contracts:
        assert _sha256(ROOT / item["path"]) == item["sha256"]

    assert contracts[0]["decision"] == "already_frozen"
    assert all(item["decision"] == "approved" for item in contracts[1:])
    assert all("reviewed_sha256" in item for item in contracts[1:])
    assert all(item["owner_action"] == "none" for item in contracts[1:])
    for item in contracts[1:]:
        contract = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
        assert contract["status"] == "frozen"
        assert contract["rules"]["candidate_is_frozen_contract"] is True
