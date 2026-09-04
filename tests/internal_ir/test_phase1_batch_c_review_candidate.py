from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts/ir-phase1-batch-c-review-candidate.json"
SUCCESSOR = ROOT / "contracts/ir-phase1-batch-f-performance-remediation.json"
REMEDIATION_SUCCESSOR = (
    ROOT / "contracts/ir-phase2-batch-a-performance-remediation-artifact-successor.json"
)
SUCCESS_CACHE_ATTESTATION = (
    ROOT
    / "tests/fixtures/internal_ir/phase1_import_success_cache_successor_candidate.json"
)


def _candidate() -> dict[str, object]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_c_candidate_binds_authorization_and_artifacts() -> None:
    candidate = _candidate()
    authorization = candidate["authorization"]
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    successors = successor["successors"]
    remediation = json.loads(REMEDIATION_SUCCESSOR.read_text(encoding="utf-8"))
    remediation_transitions = remediation["candidates"][
        str(CANDIDATE.relative_to(ROOT))
    ]["artifact_transitions"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    for section in ("implementation_artifacts", "test_artifacts"):
        for relative_path, expected_hash in candidate[section].items():
            actual_hash = _sha256(ROOT / relative_path)
            if actual_hash == expected_hash:
                continue
            amendment = remediation_transitions.get(relative_path)
            if amendment is not None:
                assert amendment["predecessor_sha256"] == expected_hash
                assert amendment["successor_sha256"] == actual_hash
                continue
            amendment = successors[relative_path]
            if (
                relative_path == "flagquantum/_compiler/importers/circuit_ir.py"
                and actual_hash != amendment["current_sha256"]
            ):
                successor_attestation = json.loads(
                    SUCCESS_CACHE_ATTESTATION.read_text(encoding="utf-8")
                )
                implementation = successor_attestation["implementation"]
                assert (
                    implementation["predecessor_sha256"] == amendment["current_sha256"]
                )
                assert implementation["successor_sha256"] == actual_hash
                assert implementation["successor_sha256_mode"] == "exact"
                continue
            assert amendment["previous_sha256"] == expected_hash
            assert amendment["current_sha256"] == actual_hash
            assert amendment["semantic_output_change"] is False


def test_batch_c_successor_is_explicitly_authorized_and_non_retroactive() -> None:
    successor = json.loads(SUCCESSOR.read_text(encoding="utf-8"))
    authorization = successor["authorization"]
    historical = successor["historical_candidate"]

    assert _sha256(ROOT / authorization["path"]) == authorization["sha256"]
    assert _sha256(ROOT / historical["path"]) == historical["sha256"]
    assert historical["mutated"] is False
    assert successor["budget"]["changed"] is False


def test_batch_c_candidate_records_importer_scope() -> None:
    candidate = _candidate()

    assert candidate["status"] == "ready_for_batch_c_review"
    assert candidate["scope"]["canonical_opcodes_imported"] == 35
    assert candidate["scope"]["phase0_manifest_status_matched"] is True
    assert candidate["scope"]["input_mutation"] is False
    assert candidate["scope"]["public_exports_added"] == []
    assert candidate["scope"]["default_path_enabled"] is False
    assert candidate["scope"]["exporter_implemented"] is False


def test_batch_c_candidate_keeps_exporter_and_later_phases_closed() -> None:
    candidate = _candidate()
    decision = candidate["exit_decision"]

    assert decision["batch_c_owner_accepted"] is False
    assert decision["batch_d_authorized"] is False
    assert decision["phase1_complete"] is False
    assert decision["phase2_authorized"] is False
    assert candidate["next_review_command"] == (
        "approve IR-PHASE1-BATCH-C-EXIT-BATCH-D"
    )
    assert (
        "No provider codegen, public API, default runtime"
        in candidate["next_review_semantics"]
    )
