from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

import flagquantum.experimental.artifacts as artifacts
from flagquantum.compiler.artifact_compilation import (
    compile_circuit_artifact_for_target,
)
from flagquantum.compiler.compilation_evidence import (
    build_compilation_evidence_bundle,
)
from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.core import _artifacts
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.experimental.artifacts import (
    CompilationEvidence,
    ProgramArtifact,
    dump_compilation_evidence,
    dump_program_artifact,
    load_compilation_evidence,
    load_program_artifact,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)


def _snapshot() -> TargetCapabilitySnapshot:
    scope = CapabilityScope(device_ids=("preview:0",))
    source = FactSource(kind="phase51_test", ref="phase51-evidence")
    values: dict[str, object] = {
        "qubits.logical_capacity": 3,
        "limits.maximum_program_operations": 4096,
        "precision.effective_dtype": "complex128",
        "gates.native": ("h", "cx", "swap"),
        "measurements.results": ("samples",),
        "artifacts.profiles": ("openqasm-3.0",),
        "limits.maximum_shots": 4096,
    }
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase51-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase51-environment",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=tuple(
            CapabilityFact(
                name=name,
                value=value,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=(
                    FactExposure.OBSERVED
                    if name == "precision.effective_dtype"
                    else FactExposure.DECLARED
                ),
                source=source,
            )
            for name, value in values.items()
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="phase51-evidence",
                sha256="a" * 64,
                level=EvidenceLevel.OBSERVABLE,
                scope=scope,
            ),
        ),
    )


def _source() -> _artifacts.ProgramArtifactV2:
    return _artifacts.ProgramArtifactV2.from_circuit_ir(
        CircuitIR(
            2,
            (Instruction("h", (0,)), Instruction("cx", (0, 1))),
            dtype="complex128",
            measurements=(MeasurementNode("samples", (0, 1)),),
        ),
        producer="phase51-source",
    )


def _compiled_versions() -> tuple[object, object, object, object]:
    source = _source()
    snapshot = _snapshot()
    v1_result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="phase51-compiler",
        evaluated_at=_NOW,
    )
    v2_result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="phase51-compiler",
        evaluated_at=_NOW,
        coupling_map=DirectedCouplingMap(2, ((0, 1),)),
    )
    v3_result = compile_circuit_artifact_for_target(
        source,
        backend="qasm",
        profile="openqasm-3.0",
        snapshot=snapshot,
        producer="phase51-compiler",
        evaluated_at=_NOW,
        coupling_map=DirectedCouplingMap(3, ((0, 1), (1, 2))),
        initial_layout=(0, 2),
    )
    evidence = tuple(
        build_compilation_evidence_bundle(
            result,
            snapshot=snapshot,
            producer="phase51-compiler",
        )
        for result in (v1_result, v2_result, v3_result)
    )
    return (
        v1_result.executable_artifact,
        v2_result.executable_artifact,
        v3_result.executable_artifact,
        evidence,
    )


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def test_preview_round_trips_all_artifact_and_evidence_versions() -> None:
    artifact_v2a, artifact_v2b, artifact_v3, evidence = _compiled_versions()
    legacy = _artifacts.ProgramArtifact(
        kind=_artifacts.ArtifactKind.LOGICAL,
        payload={"name": "legacy"},
        producer="phase51-legacy",
    )
    serialized_artifacts = (
        _canonical(legacy.to_dict()),
        artifact_v2a.to_json(),
        artifact_v2b.to_json(),
        artifact_v3.to_json(),
    )
    for serialized in serialized_artifacts:
        view = load_program_artifact(serialized)
        assert isinstance(view, ProgramArtifact)
        assert dump_program_artifact(view) == serialized
        assert view.to_json() == serialized
        assert len(view.identity) == 64
        assert view.kind in {"logical", "executable"}

    assert tuple(item.version for item in evidence) == ("1.0", "2.0", "3.0")
    for item in evidence:
        serialized = item.to_json()
        view = load_compilation_evidence(serialized)
        assert isinstance(view, CompilationEvidence)
        assert view.version == item.version
        assert view.identity == item.bundle_identity
        assert dump_compilation_evidence(view) == serialized
        assert view.to_json() == serialized


def test_preview_views_are_frozen_and_dumpers_reject_lookalikes() -> None:
    artifact, _, _, evidence = _compiled_versions()
    artifact_view = load_program_artifact(artifact.to_json())
    evidence_view = load_compilation_evidence(evidence[0].to_json())
    artifact_dict = artifact_view.to_dict()
    evidence_dict = evidence_view.to_dict()
    artifact_dict["producer"] = "changed-copy"
    evidence_dict["producer"] = "changed-copy"
    assert artifact_view.producer == "phase51-compiler"
    assert evidence_view.producer == "phase51-compiler"
    with pytest.raises(FrozenInstanceError):
        artifact_view._value = object()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evidence_view._value = object()  # type: ignore[misc]
    with pytest.raises(TypeError, match="ProgramArtifact view"):
        dump_program_artifact(artifact)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CompilationEvidence view"):
        dump_compilation_evidence(evidence[0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="loaded Core artifact"):
        ProgramArtifact({"version": "2.0"})
    with pytest.raises(TypeError, match="loaded Core evidence"):
        CompilationEvidence({"version": "1.0"})


def test_preview_loaders_fail_closed_and_are_lazy() -> None:
    assert artifacts.__all__ == (
        "CompilationEvidence",
        "ProgramArtifact",
        "dump_compilation_evidence",
        "dump_program_artifact",
        "load_compilation_evidence",
        "load_program_artifact",
    )
    with pytest.raises(TypeError, match="JSON text"):
        load_program_artifact(b"{}")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate program artifact field"):
        load_program_artifact('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="unsupported artifact envelope version"):
        load_program_artifact(
            _canonical(
                {
                    "schema": "flagquantum.program_artifact",
                    "version": "99.0",
                    "kind": "logical",
                    "producer": "future",
                    "required_capabilities": [],
                    "parent_hashes": [],
                    "payload": {},
                    "metadata": {},
                }
            )
        )
    with pytest.raises(ValueError, match="unsupported compilation evidence"):
        load_compilation_evidence('{"version":"99.0"}')
    with pytest.raises(ValueError, match="maximum UTF-8 bytes"):
        load_program_artifact(" " * (18 * 1024 * 1024 + 1))

    script = (
        "import sys, flagquantum, flagquantum.experimental; "
        "print('flagquantum.experimental.artifacts' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"
