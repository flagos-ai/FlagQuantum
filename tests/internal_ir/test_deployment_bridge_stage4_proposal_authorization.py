from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/deployment-bridge-stage3-exit-stage4-proposal-authorization.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_proposal_authorization_binds_owner_review() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))

    assert authorization["status"] == "approved"
    assert authorization["approval_command"] == (
        "approve DEPLOYMENT-BRIDGE-STAGE3-EXIT-STAGE4-PROPOSAL"
    )
    candidate = authorization["review_candidate"]
    assert _sha256(ROOT / candidate["path"]) == candidate["sha256"]


def test_stage4_authorization_is_proposal_only() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    decisions = authorization["decisions"]
    scope = authorization["allowed_scope"]

    allowed = {
        "stage3_owner_accepted",
        "stage3_exit_authorized",
        "stage4_proposal_authorized",
    }
    assert all(decisions[name] is True for name in allowed)
    assert all(
        value is False for name, value in decisions.items() if name not in allowed
    )
    assert scope["proposal_and_offline_contract_tests_only"] is True
    assert set(scope["roots"]) == {
        "docs/development",
        "contracts",
        "tests/internal_ir",
    }
