from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.artifact_profiles import (
    ARTIFACT_PROFILE_REGISTRY,
    ArtifactEncoding,
    artifact_profile_definition,
)
from flagquantum._compiler.executable_artifact import (
    ArtifactSealStatus,
    SealedExecutableArtifact,
    seal_executable_artifact,
    verify_executable_artifact,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_ir import TargetIR, TargetOperation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/internal_ir/phase3_batch_c_artifacts.json"
COMPILATION_IDENTITY = "c" * 64


def _records() -> list[dict[str, object]]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]


def _case(record: dict[str, object]):
    profile = ArtifactProfile(
        ArtifactFormat(record["format"]),
        record["version"],
    )
    target = TargetCapabilities(
        target_class=TargetClass(record["target_class"]),
        logical_qubit_capacity=2,
        physical_qubit_capacity=2,
        native_gates=(GateCapability("rx"), GateCapability("cx")),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(profile,),
        topology=DirectedCouplingGraph(2, ((0, 1), (1, 0))),
        maximum_shots=100,
        maximum_program_operations=100,
    )
    required_results = tuple(
        MeasurementResult(item) for item in record["required_results"]
    )
    target_ir = TargetIR(
        "a" * 64,
        target.semantic_fingerprint,
        (0, 1),
        (
            TargetOperation("rx", (0,), {"theta": 0.25}),
            TargetOperation("cx", (0, 1)),
        ),
        required_results,
        record["requested_shots"],
    )
    return profile, target, target_ir, record["payload_utf8"].encode("utf-8")


def test_closed_registry_has_exact_format_version_and_media_semantics() -> None:
    assert len(ARTIFACT_PROFILE_REGISTRY) == 4
    for record in _records():
        profile, _, _, _ = _case(record)
        definition = artifact_profile_definition(profile)
        assert definition.profile == profile
        assert definition.encoding is ArtifactEncoding.UTF8_TEXT
        assert definition.portable is True
        assert definition.media_type
    with pytest.raises(TypeError):
        ARTIFACT_PROFILE_REGISTRY[
            ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0")
        ] = object()


def test_anonymous_artifacts_seal_verify_and_match_golden_identities() -> None:
    for record in _records():
        profile, target, target_ir, payload = _case(record)
        result = seal_executable_artifact(
            target_ir,
            target,
            profile,
            payload,
            compilation_identity=COMPILATION_IDENTITY,
        )

        assert result.ok and result.artifact is not None
        assert result.status is ArtifactSealStatus.SEALED
        assert result.artifact.payload_content_hash == record["expected_payload_hash"]
        assert result.artifact.artifact_identity == record["expected_artifact_identity"]
        assert verify_executable_artifact(result.artifact, target_ir, target).ok


def test_payload_profile_target_and_identity_mismatch_fail_closed() -> None:
    profile, target, target_ir, payload = _case(_records()[1])
    wrong_payload = payload.replace(b"rx(0.25)", b"rx(0.5)")
    mismatched_payload = seal_executable_artifact(
        target_ir,
        target,
        profile,
        wrong_payload,
        compilation_identity=COMPILATION_IDENTITY,
    )
    other_profile = ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0")
    mismatched_profile = seal_executable_artifact(
        target_ir,
        target,
        other_profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    )
    mismatched_target = seal_executable_artifact(
        target_ir,
        replace(target, supports_noise=True),
        profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    )
    invalid_identity = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity="not-a-digest",
    )

    assert all(
        not result.ok and result.artifact is None
        for result in (
            mismatched_payload,
            mismatched_profile,
            mismatched_target,
            invalid_identity,
        )
    )


def test_post_seal_payload_metadata_and_target_tampering_are_rejected() -> None:
    profile, target, target_ir, payload = _case(_records()[2])
    sealed = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    ).artifact
    assert sealed is not None

    payload_tampered = replace(sealed, payload=sealed.payload + b"\n")
    media_tampered = replace(sealed, media_type="application/octet-stream")
    identity_tampered = replace(sealed, compilation_identity="d" * 64)
    target_tampered = replace(target_ir, requested_shots=1)

    for artifact, candidate_ir in (
        (payload_tampered, target_ir),
        (media_tampered, target_ir),
        (identity_tampered, target_ir),
        (sealed, target_tampered),
    ):
        result = verify_executable_artifact(artifact, candidate_ir, target)
        assert not result.ok
        assert result.artifact is None


def test_payload_is_copied_as_immutable_bytes_and_artifact_is_frozen() -> None:
    profile, target, target_ir, payload = _case(_records()[3])
    source = bytearray(payload)
    rejected = seal_executable_artifact(
        target_ir,
        target,
        profile,
        source,
        compilation_identity=COMPILATION_IDENTITY,
    )
    sealed = seal_executable_artifact(
        target_ir,
        target,
        profile,
        bytes(source),
        compilation_identity=COMPILATION_IDENTITY,
    ).artifact

    assert not rejected.ok
    assert sealed is not None
    source.extend(b"secret")
    assert sealed.payload == payload
    with pytest.raises(FrozenInstanceError):
        sealed.payload = b"changed"


def test_artifact_schema_and_payload_reject_execution_secrets() -> None:
    forbidden = {
        "provider",
        "backend_id",
        "credential",
        "token",
        "url",
        "job_id",
        "queue",
        "dispatch_locator",
    }
    assert {item.name for item in fields(SealedExecutableArtifact)}.isdisjoint(
        forbidden
    )
    record = copy.deepcopy(_records()[0])
    profile, target, target_ir, _ = _case(record)
    payload = json.dumps(
        {**target_ir.canonical(), "credential": "secret"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    result = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    )
    assert not result.ok


def test_artifact_identity_changes_across_every_semantic_boundary() -> None:
    identities = []
    for record in _records():
        profile, target, target_ir, payload = _case(record)
        sealed = seal_executable_artifact(
            target_ir,
            target,
            profile,
            payload,
            compilation_identity=COMPILATION_IDENTITY,
        ).artifact
        assert sealed is not None
        identities.append(sealed.artifact_identity)
    assert len(set(identities)) == 4

    profile, target, target_ir, payload = _case(_records()[1])
    first = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity=COMPILATION_IDENTITY,
    ).artifact
    changed = seal_executable_artifact(
        target_ir,
        target,
        profile,
        payload,
        compilation_identity="d" * 64,
    ).artifact
    assert first is not None and changed is not None
    assert first.artifact_identity != changed.artifact_identity


def test_artifact_identity_ignores_python_hash_seed() -> None:
    script = f"""
import json
from pathlib import Path
from tests.internal_ir.test_phase3_executable_artifact import _case
from flagquantum._compiler.executable_artifact import seal_executable_artifact
record = json.loads(Path({str(FIXTURES)!r}).read_text())[\"fixtures\"][1]
profile, target, target_ir, payload = _case(record)
result = seal_executable_artifact(target_ir, target, profile, payload, compilation_identity={'c' * 64!r})
print(result.artifact.artifact_identity)
"""
    values = []
    for seed in (1, 8675309):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = str(seed)
        values.append(
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
    assert values[0] == values[1]


def test_executable_artifact_remains_private() -> None:
    for name in (
        "ArtifactProfileDefinition",
        "SealedExecutableArtifact",
        "seal_executable_artifact",
        "verify_executable_artifact",
    ):
        assert not hasattr(fq, name)
