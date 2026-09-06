from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REVIEW = (
    ROOT / "contracts/deployment-bridge-stage7-quafu-selection-review-candidate.json"
)


def test_quafu_selection_review_is_blocked_pending_target_qualification() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "blocked_pending_target_qualification"
    assert review["selection"]["provider_namespace"] == "quafu"
    assert review["selection"]["sandbox_target_identity"] is None
    assert review["qualification"]["provider_verified"] is True
    assert review["qualification"]["concrete_target_verified"] is False
    assert review["qualification"]["non_billable_verified"] is False
    assert review["qualification"]["sandbox_only_verified"] is False
    assert review["blockers"]


def test_quafu_selection_review_authorizes_no_implementation_or_access() -> None:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert all(value is False for value in review["proposed_decisions"].values())
    assert review["next_owner_action"]["stage7_entry_available_now"] is False
    assert review["next_owner_action"]["required_form"].startswith(
        "approve DEPLOYMENT-BRIDGE-STAGE7-TARGET-QUALIFICATION"
    )
    assert public_api_snapshot.validate() == ()
    assert not hasattr(fq, "activate_quafu_sandbox")
