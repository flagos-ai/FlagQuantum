from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase3-entry-batch-a-review-candidate.json"
CANDIDATE_SHA256 = "3eac168d3d15051cbb19d109dd7390983445bee14927a21d9f88f185c9d6b7b3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase3_entry_candidate_preserves_review_record() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert _sha256(CANDIDATE) == CANDIDATE_SHA256
    assert candidate["status"] == "ready_for_owner_approval"
    assert all(candidate["technical_review"].values())


def test_phase3_entry_candidate_authorizes_nothing_before_exact_approval() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert all(value is False for value in candidate["authorization"].values())
    assert all(value is False for value in candidate["formal_signoffs"].values())
    assert candidate["approval_command"] == "approve IR-PHASE3-ENTRY-BATCH-A"


def test_phase3_batch_a_scope_is_private_provider_free_and_sequential() -> None:
    scope = json.loads(CANDIDATE.read_text(encoding="utf-8"))["batch_a_scope"]

    assert scope["profile"] == "private_provider_free_target_capabilities_v1"
    assert scope["allowed_root"] == "flagquantum/_compiler"
    assert scope["public_exports"] == []
    assert scope["default_path_enabled"] is False
    assert scope["real_provider_or_backend_identity_allowed"] is False
    assert scope["later_batches_authorized"] is False
