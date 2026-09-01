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


def test_review_packet_binds_exact_contracts_without_implying_approval() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))

    assert packet["status"] == "ready_for_owner_review"
    assert packet["approval_record"] == {
        "approved_by": None,
        "approved_at": None,
        "approved_proposals": [],
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
    assert all(item["decision"] == "pending" for item in contracts[1:])
