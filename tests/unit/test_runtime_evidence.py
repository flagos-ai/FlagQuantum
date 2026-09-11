import json
from dataclasses import FrozenInstanceError

import pytest

from benchmarks.audit_results import audit_paths
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    verify_evidence_artifact,
)
from tools.seal_runtime_evidence import main as seal_main

pytestmark = pytest.mark.unit

KEY = b"issue-038-test-key"


def _provenance(device_count=2) -> RuntimeProvenance:
    return RuntimeProvenance(
        commit="a" * 40,
        workload_sha256="b" * 64,
        command=("torchrun", "--nproc-per-node=2", "benchmark.py"),
        devices=tuple(f"GPU-{rank}" for rank in range(device_count)),
        topology="NV12",
        rank_mapping=tuple(f"rank{rank}=GPU-{rank}" for rank in range(device_count)),
        collective_backend="nccl",
        warmup=2,
        iterations=10,
        seeds=(7,),
        raw_log_sha256="c" * 64,
        fallback_events=(),
    )


def _measured_payload() -> dict:
    return {
        "benchmark": "signed_runtime_test",
        "claim_evidence_type": "release_payload",
        "release_payload": True,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "release_gate_allowed": True,
        "measured_peak_memory_bytes": 1024,
        "timings": {"iteration_seconds": [0.1, 0.09]},
    }


def test_artifact_classes_are_immutable_and_scopes_are_distinct():
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.DEVELOPMENT_RUN,
        evidence_scope=EvidenceScope.ONE_GPU_LOCAL,
        provenance=_provenance(1),
        evidence={"status": "executed"},
        signing_key=KEY,
    )
    with pytest.raises(FrozenInstanceError):
        artifact.schema = "edited"
    with pytest.raises(TypeError):
        artifact.evidence["status"] = "edited"
    assert {scope.value for scope in EvidenceScope} == {
        "one_gpu_local",
        "two_gpu_semantic_regression",
        "scheduled_4_8_gpu_scale",
    }


@pytest.mark.parametrize(
    "artifact_class", [ArtifactClass.PLAN, ArtifactClass.DEVELOPMENT_RUN]
)
def test_estimates_cannot_populate_measured_fields(artifact_class):
    with pytest.raises(ValueError, match="cannot populate measured fields"):
        create_evidence_artifact(
            artifact_class=artifact_class,
            evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
            provenance=_provenance(),
            evidence={"measured_peak_memory_bytes": 4096},
            signing_key=KEY,
        )


def test_signed_runtime_artifact_rejects_editing():
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.SCHEDULED_SCALE,
        provenance=_provenance(4),
        evidence=_measured_payload(),
        signing_key=KEY,
    )
    payload = artifact.summary()
    assert verify_evidence_artifact(payload, signing_key=KEY) == (True, ())

    payload["evidence"]["measured_peak_memory_bytes"] = 2048
    valid, errors = verify_evidence_artifact(payload, signing_key=KEY)
    assert valid is False
    assert "runtime evidence content checksum mismatch" in errors


def test_release_scanner_requires_signed_runtime_envelope(tmp_path):
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.SCHEDULED_SCALE,
        provenance=_provenance(4),
        evidence=_measured_payload(),
        signing_key=KEY,
    )
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(artifact.summary()), encoding="utf-8")

    wrong_key = audit_paths([path], require_scalability=True, signing_key=b"wrong")
    assert wrong_key["claimable_count"] == 0
    assert wrong_key["records"][0]["status"] == "provenance_rejected"

    signed = audit_paths([path], require_scalability=True, signing_key=KEY)
    assert signed["records"][0]["status"] == "ok"


def test_runtime_sealer_refuses_to_emit_without_protected_key(tmp_path, monkeypatch):
    measurements = tmp_path / "measurements.json"
    measurements.write_text("{}", encoding="utf-8")
    raw_log = tmp_path / "run.log"
    raw_log.write_text("executed", encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("FQ_EVIDENCE_SIGNING_KEY", raising=False)

    with pytest.raises(SystemExit, match="FQ_EVIDENCE_SIGNING_KEY is not configured"):
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--raw-log",
                str(raw_log),
                "--workload",
                str(workload),
                "--output",
                str(tmp_path / "sealed.json"),
                "--scope",
                "two_gpu_semantic_regression",
                "--command",
                "torchrun benchmark.py",
                "--collective-backend",
                "nccl",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--world-size",
                "2",
                "--seed",
                "7",
            ]
        )
