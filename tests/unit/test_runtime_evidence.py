import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from benchmarks.audit_results import audit_paths
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    verify_evidence_artifact,
)
from tools.seal_runtime_evidence import _rank_devices
from tools.seal_runtime_evidence import main as seal_main

pytestmark = pytest.mark.unit

KEY = b"issue-038-test-key"


def test_declared_rank_devices_can_name_a_second_host():
    """A rank on another host must be recorded as such, not as a local device."""

    devices, mapping = _rank_devices(
        ("GPU-aaa@jp-a800-172", "GPU-bbb@jp-a800-171"), world_size=2
    )

    assert devices == ("GPU-aaa", "GPU-bbb")
    assert mapping == (
        "rank=0:node=jp-a800-172:device_uuid=GPU-aaa",
        "rank=1:node=jp-a800-171:device_uuid=GPU-bbb",
    )


def test_declared_rank_devices_must_cover_every_rank():
    with pytest.raises(SystemExit, match="received 1 device UUIDs"):
        _rank_devices(("GPU-aaa",), world_size=2)


def test_undeclared_rank_devices_still_map_ranks_positionally(monkeypatch):
    monkeypatch.setattr(
        "tools.seal_runtime_evidence._output",
        lambda command: "GPU-aaa\nGPU-bbb\nGPU-ccc\n",
    )

    devices, mapping = _rank_devices(None, world_size=2)

    assert devices == ("GPU-aaa", "GPU-bbb")
    assert mapping == (
        "rank=0:device_uuid=GPU-aaa",
        "rank=1:device_uuid=GPU-bbb",
    )


def test_undeclared_rank_devices_fail_closed_when_the_host_is_too_small(monkeypatch):
    monkeypatch.setattr(
        "tools.seal_runtime_evidence._output", lambda command: "GPU-aaa\n"
    )

    with pytest.raises(SystemExit, match="requested world size 4, detected 1 GPUs"):
        _rank_devices(None, world_size=4)


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
        "multi_node_16_gpu_scale",
    }


def test_multi_node_scale_carries_sixteen_devices_and_nothing_nearer():
    """The widest scope states the one device count it was added to carry.

    Sixteen devices is two full hosts of the measured cluster, and it is the
    smallest world in which the frozen MPS capacity workload fits. The scope
    must therefore accept exactly that count and refuse its neighbours, so a
    fifteen- or seventeen-device provenance record cannot be sealed as if it had
    crossed the node boundary at the frozen topology.
    """

    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.MULTI_NODE_SCALE,
        provenance=_provenance(16),
        evidence=_measured_payload(),
        signing_key=KEY,
    )
    payload = artifact.summary()
    assert payload["evidence_scope"] == "multi_node_16_gpu_scale"
    assert len(payload["provenance"]["devices"]) == 16
    assert verify_evidence_artifact(payload, signing_key=KEY) == (True, ())

    for device_count in (1, 2, 4, 8, 15, 17):
        with pytest.raises(ValueError, match="requires device count"):
            create_evidence_artifact(
                artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
                evidence_scope=EvidenceScope.MULTI_NODE_SCALE,
                provenance=_provenance(device_count),
                evidence=_measured_payload(),
                signing_key=KEY,
            )


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


def test_keyless_evidence_has_only_a_checksum_and_no_authentication_claim() -> None:
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
        provenance=_provenance(),
        evidence=_measured_payload(),
    ).summary()
    assert artifact["schema"] == "flagquantum_runtime_evidence_v2"
    assert set(artifact["integrity"]) == {"algorithm", "content_sha256"}
    assert artifact["integrity"]["algorithm"] == "sha256"
    assert verify_evidence_artifact(artifact) == (True, ())


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("evidence", "measured_peak_memory_bytes", 8192),
        ("provenance", "commit", "d" * 40),
        ("provenance", "workload_sha256", "d" * 64),
        ("provenance", "raw_log_sha256", "d" * 64),
        ("provenance", "devices", ["GPU-other", "GPU-1"]),
    ],
)
def test_keyless_evidence_rejects_changes_to_measurements_and_provenance(
    section: str, field: str, value: object
) -> None:
    payload = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
        provenance=_provenance(),
        evidence=_measured_payload(),
    ).summary()
    payload[section][field] = value
    valid, errors = verify_evidence_artifact(payload)
    assert not valid
    assert "runtime evidence content checksum mismatch" in errors


def test_keyless_verification_cannot_downgrade_a_legacy_signature() -> None:
    payload = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
        provenance=_provenance(),
        evidence=_measured_payload(),
        signing_key=KEY,
    ).summary()
    assert verify_evidence_artifact(payload, signing_key=KEY) == (True, ())
    assert "runtime evidence signature mismatch" in verify_evidence_artifact(payload)[1]
    payload["integrity"]["algorithm"] = "sha256"
    assert "algorithm does not match" in "; ".join(
        verify_evidence_artifact(payload, signing_key=KEY)[1]
    )


@pytest.mark.parametrize("algorithm", ["hmac-sha256", "unknown"])
def test_keyless_schema_rejects_a_different_algorithm(algorithm: str) -> None:
    payload = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
        provenance=_provenance(),
        evidence=_measured_payload(),
    ).summary()
    payload["integrity"]["algorithm"] = algorithm
    assert not verify_evidence_artifact(payload)[0]


@pytest.mark.parametrize("key", [KEY, b""])
def test_release_scanner_checks_envelope_integrity(tmp_path: Path, key: bytes) -> None:
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.SCHEDULED_SCALE,
        provenance=_provenance(4),
        evidence=_measured_payload(),
        signing_key=key,
    )
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(artifact.summary()), encoding="utf-8")

    if key:
        wrong_key = audit_paths([path], require_scalability=True, signing_key=b"wrong")
        assert wrong_key["claimable_count"] == 0
        assert wrong_key["records"][0]["status"] == "provenance_rejected"

    signed = audit_paths([path], require_scalability=True, signing_key=key)
    assert signed["records"][0]["status"] == "ok"


def test_runtime_sealer_emits_keyless_evidence_with_complete_recorded_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps(_measured_payload()), encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(asdict(_provenance())), encoding="utf-8")
    output = tmp_path / "sealed.json"
    monkeypatch.delenv("FQ_EVIDENCE_SIGNING_KEY", raising=False)

    assert (
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--recorded-provenance",
                str(provenance),
                "--output",
                str(output),
                "--scope",
                "two_gpu_semantic_regression",
                "--world-size",
                "2",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["schema"] == "flagquantum_runtime_evidence_v2"
    assert payload["provenance"] == json.loads(provenance.read_text())
    assert verify_evidence_artifact(payload) == (True, ())


def test_runtime_sealer_requires_a_complete_provenance_source(tmp_path, monkeypatch):
    """A fresh seal must state where its provenance comes from, either way."""

    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"world_size": 2}), encoding="utf-8")
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEY.decode())

    with pytest.raises(SystemExit, match="sealing a fresh run requires"):
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--output",
                str(tmp_path / "sealed.json"),
                "--scope",
                "two_gpu_semantic_regression",
                "--world-size",
                "2",
            ]
        )


def test_runtime_sealer_accepts_a_declared_inventory_from_a_second_host(
    tmp_path, monkeypatch
):
    """A world this host cannot see must still seal when the caller declared it.

    The preflight asks whether the invocation has the devices its world size
    claims. Only the caller knows the other host's devices, so counting what
    this process can detect would refuse every multi-node seal after the run had
    already been measured. Detection is stubbed to one device here, which is the
    situation on each of the two hosts that ran the sixteen ranks.
    """

    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEY.decode())
    monkeypatch.setattr("tools.seal_runtime_evidence._output", lambda command: "a" * 40)
    measurements = tmp_path / "measurements.json"
    measurements.write_text(
        json.dumps(dict(_measured_payload(), world_size=2)), encoding="utf-8"
    )
    raw_log = tmp_path / "run.log"
    raw_log.write_text("executed", encoding="utf-8")
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    destination = tmp_path / "sealed.json"

    assert (
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--raw-log",
                str(raw_log),
                "--workload",
                str(workload),
                "--output",
                str(destination),
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
                "--device-uuid",
                "GPU-aaa@host-a",
                "--device-uuid",
                "GPU-bbb@host-b",
            ]
        )
        == 0
    )

    sealed = json.loads(destination.read_text(encoding="utf-8"))
    assert sealed["provenance"]["devices"] == ["GPU-aaa", "GPU-bbb"]
    assert sealed["provenance"]["rank_mapping"] == [
        "rank=0:node=host-a:device_uuid=GPU-aaa",
        "rank=1:node=host-b:device_uuid=GPU-bbb",
    ]
    valid, errors = verify_evidence_artifact(sealed, signing_key=KEY)
    assert valid, errors


def test_runtime_sealer_reseal_reuses_the_recorded_provenance(tmp_path, monkeypatch):
    """A re-seal states the run it re-seals, not the checkout doing the re-seal.

    The sealer detects a fresh provenance from this host. Re-sealing an already
    measured payload must not restamp it with this checkout's commit, this host's
    devices, this host's topology or this host's raw-log digest, because the
    payload would then describe a run that never happened.
    """

    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEY.decode())
    recorded = _provenance(2)
    original = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope.TWO_GPU_SEMANTIC,
        provenance=recorded,
        evidence=_measured_payload(),
        signing_key=KEY,
    )
    source = tmp_path / "measured.json"
    source.write_text(json.dumps(original.summary()), encoding="utf-8")

    measurements = tmp_path / "measurements.json"
    measurements.write_text(
        json.dumps(dict(original.summary()["evidence"], world_size=2)),
        encoding="utf-8",
    )
    destination = tmp_path / "resealed.json"
    assert (
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--output",
                str(destination),
                "--scope",
                "two_gpu_semantic_regression",
                "--world-size",
                "2",
                "--recorded-provenance",
                str(source),
                "--release-payload",
            ]
        )
        == 0
    )

    resealed = json.loads(destination.read_text(encoding="utf-8"))
    # The sealed document round-trips through JSON, so tuple-valued provenance
    # fields come back as lists; compare against the round-tripped original.
    assert (
        resealed["provenance"]
        == json.loads(json.dumps(original.summary()))["provenance"]
    )
    assert resealed["provenance"]["commit"] == recorded.commit
    assert resealed["evidence"]["release_gate_allowed"] is True
    valid, errors = verify_evidence_artifact(resealed, signing_key=KEY)
    assert valid, errors


def test_runtime_sealer_reseal_rejects_an_incomplete_record(tmp_path, monkeypatch):
    """A recorded provenance missing a field fails closed rather than being filled."""

    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEY.decode())
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"world_size": 2}), encoding="utf-8")
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"commit": "a" * 40}), encoding="utf-8")

    with pytest.raises(SystemExit, match="is missing 'workload_sha256'"):
        seal_main(
            [
                "--measurements",
                str(measurements),
                "--output",
                str(tmp_path / "sealed.json"),
                "--scope",
                "two_gpu_semantic_regression",
                "--world-size",
                "2",
                "--recorded-provenance",
                str(partial),
            ]
        )
