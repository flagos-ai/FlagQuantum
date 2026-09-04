from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = (
    ROOT / "contracts/ir-phase1-import-success-cache-successor-authorization.json"
)
REVIEW = (
    ROOT / "contracts/ir-phase1-import-success-cache-successor-review-candidate.json"
)
ATTESTATION = (
    ROOT
    / "tests/fixtures/internal_ir/phase1_import_success_cache_successor_candidate.json"
)
IMPORTER = ROOT / "flagquantum/_compiler/importers/circuit_ir.py"
OLD_IMPORTER_SHA256 = "e67ac3e5dad0cc3e5241708c6b5592bc945595870d637c1194a41172fbdc23c3"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authorization_is_non_retroactive_and_binds_all_importer_history() -> None:
    authorization = _load(AUTHORIZATION)

    assert authorization["status"] == "approved_pending_compiler_successor"
    assert _sha256(REVIEW) == authorization["review_candidate"]["sha256"]
    assert authorization["historical_records_mutated"] is False
    assert authorization["predecessor_importer_sha256"] == OLD_IMPORTER_SHA256
    for artifact in authorization["historical_bindings"]:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert authorization["successor_attestation"]["path"] == str(
        ATTESTATION.relative_to(ROOT)
    )
    assert (
        authorization["successor_attestation"]["sha256_mode"] == "exact_at_submission"
    )
    assert authorization["successor_attestation"]["wildcards_allowed"] is False


def test_authorization_scope_preserves_semantics_and_existing_gate() -> None:
    authorization = _load(AUTHORIZATION)
    allowed = authorization["allowed_change"]
    prohibited = authorization["prohibited_changes"]

    assert allowed["change_class"] == "successful_result_cache"
    assert allowed["cache_successes_only"] is True
    assert allowed["cache_failures"] is False
    assert allowed["implementation_path"] == str(IMPORTER.relative_to(ROOT))
    assert prohibited["performance_budget"] is True
    assert prohibited["measurement_method_or_statistics"] is True
    assert prohibited["public_api"] is True
    assert prohibited["default_compilation_semantics"] is True
    assert prohibited["test_wrapper_or_benchmark_cache"] is True


def test_current_importer_is_old_or_has_one_exact_authorized_successor() -> None:
    authorization = _load(AUTHORIZATION)
    actual_hash = _sha256(IMPORTER)

    if actual_hash == OLD_IMPORTER_SHA256:
        assert not ATTESTATION.exists()
        return

    attestation = _load(ATTESTATION)
    implementation = attestation["implementation"]
    assert _sha256(AUTHORIZATION) == attestation["authorization"]["sha256"]
    assert implementation["path"] == str(IMPORTER.relative_to(ROOT))
    assert implementation["predecessor_sha256"] == OLD_IMPORTER_SHA256
    assert implementation["successor_sha256_mode"] == "exact"
    assert re.fullmatch(r"[0-9a-f]{64}", implementation["successor_sha256"])
    assert "*" not in implementation["successor_sha256"]
    assert implementation["successor_sha256"] == actual_hash
    assert implementation["cache_location"] == "product_importer_module"
    assert implementation["test_wrapper_cache_used"] is False

    guards = attestation["semantic_guards"]
    assert guards["successful_results_only"] is True
    assert guards["failures_never_cached"] is True
    assert guards["nested_content_change_reimports_and_revalidates"] is True
    assert guards["trainable_tensor_identity_change_reimports_and_revalidates"] is True
    assert guards["public_api_changed"] is False
    assert guards["default_compilation_semantics_changed"] is False

    budget = attestation["performance_budget"]
    assert budget["changed"] is False
    assert budget["measurement_method_changed"] is False
    assert budget["statistics_changed"] is False
    assert _sha256(ROOT / budget["path"]) == budget["sha256"]

    required_tests = set(authorization["required_evidence_tests"])
    assert required_tests <= set(attestation["evidence_artifacts"])
    for relative_path, expected_hash in attestation["evidence_artifacts"].items():
        assert _sha256(ROOT / relative_path) == expected_hash
