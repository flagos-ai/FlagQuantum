from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.executable_artifact import seal_executable_artifact
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.provider_conformance import (
    ConformanceCase,
    ConformanceStatus,
    ConformanceTargetFamily,
    LocalConformanceDriver,
    OfflineConformanceDriver,
    ProviderExtension,
    ProviderExtensionEntry,
    run_conformance_case,
)
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
ARTIFACT_FIXTURES = ROOT / "tests/fixtures/internal_ir/phase3_batch_c_artifacts.json"
CONFORMANCE_FIXTURES = (
    ROOT / "tests/fixtures/internal_ir/phase3_batch_e_conformance.json"
)
ADAPTER_IDENTITY = "d" * 64
COMPILATION_IDENTITY = "c" * 64
RESULT_PAYLOAD = b"flagquantum-conformance-result-v1"


def _artifact_records() -> dict[str, dict[str, object]]:
    payload = json.loads(ARTIFACT_FIXTURES.read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload["fixtures"]}


def _conformance_records() -> list[dict[str, object]]:
    return json.loads(CONFORMANCE_FIXTURES.read_text(encoding="utf-8"))["cases"]


def _case(record: dict[str, object]) -> ConformanceCase:
    artifact_record = _artifact_records()[record["artifact_fixture"]]
    profile = ArtifactProfile(
        ArtifactFormat(artifact_record["format"]),
        artifact_record["version"],
    )
    target = TargetCapabilities(
        target_class=TargetClass(artifact_record["target_class"]),
        logical_qubit_capacity=2,
        physical_qubit_capacity=2,
        native_gates=(GateCapability("rx"), GateCapability("cx")),
        measurement_results=(MeasurementResult.STATE,),
        artifact_profiles=(profile,),
        topology=DirectedCouplingGraph(2, ((0, 1), (1, 0))),
        maximum_shots=100,
        maximum_program_operations=100,
    )
    target_ir = TargetIR(
        "a" * 64,
        target.semantic_fingerprint,
        (0, 1),
        (
            TargetOperation("rx", (0,), {"theta": 0.25}),
            TargetOperation("cx", (0, 1)),
        ),
        tuple(MeasurementResult(item) for item in artifact_record["required_results"]),
        artifact_record["requested_shots"],
    )
    artifact = seal_executable_artifact(
        target_ir,
        target,
        profile,
        artifact_record["payload_utf8"].encode("utf-8"),
        compilation_identity=COMPILATION_IDENTITY,
    ).artifact
    assert artifact is not None
    extension = ProviderExtension(
        record["extension_namespace"],
        (ProviderExtensionEntry("fixture_note", record["name"]),),
    )
    return ConformanceCase(
        ConformanceTargetFamily(record["family"]),
        target,
        target_ir,
        artifact,
        (extension,),
    )


def _driver(record: dict[str, object]):
    if record["driver"] == "local_sync":
        return LocalConformanceDriver(ADAPTER_IDENTITY, RESULT_PAYLOAD)
    return OfflineConformanceDriver(ADAPTER_IDENTITY)


def _run(record: dict[str, object]):
    return run_conformance_case(
        _case(record),
        _driver(record),
        result_payload=RESULT_PAYLOAD,
    )


def test_identical_suite_passes_all_three_anonymous_target_families() -> None:
    results = [_run(record) for record in _conformance_records()]

    assert all(result.ok and result.report is not None for result in results)
    assert {result.report.family for result in results} == set(ConformanceTargetFamily)
    checks = {result.report.checks for result in results}
    assert len(checks) == 1
    for record, result in zip(_conformance_records(), results, strict=True):
        assert (
            result.report.conformance_identity
            == record["expected_conformance_identity"]
        )


def test_complete_identity_chain_is_preserved_for_every_family() -> None:
    for record in _conformance_records():
        case = _case(record)
        result = run_conformance_case(
            case,
            _driver(record),
            result_payload=RESULT_PAYLOAD,
        )
        assert result.report is not None
        report = result.report
        assert report.artifact_identity == case.artifact.artifact_identity
        assert (
            len(
                {
                    report.artifact_identity,
                    report.binding_identity,
                    report.receipt_identity,
                    report.result_identity,
                    report.conformance_identity,
                }
            )
            == 5
        )


def test_extensions_are_namespaced_immutable_and_non_semantic() -> None:
    record = _conformance_records()[1]
    with_extension = _case(record)
    without_extension = replace(with_extension, extensions=())
    first = run_conformance_case(
        with_extension,
        _driver(record),
        result_payload=RESULT_PAYLOAD,
    ).report
    second = run_conformance_case(
        without_extension,
        _driver(record),
        result_payload=RESULT_PAYLOAD,
    ).report
    assert first is not None and second is not None
    assert first.extension_namespaces == ("org.flagquantum.fixture",)
    assert second.extension_namespaces == ()
    assert first.conformance_identity == second.conformance_identity
    with pytest.raises(FrozenInstanceError):
        with_extension.extensions[0].namespace = "changed.example"


@pytest.mark.parametrize(
    ("namespace", "key"),
    [
        ("not_namespaced", "note"),
        ("org.example", "credential"),
        ("org.example", "backend_id"),
        ("org.example", "queue_depth"),
        ("org.example", "artifact_identity"),
    ],
)
def test_extensions_reject_unscoped_reserved_or_execution_sensitive_data(
    namespace: str,
    key: str,
) -> None:
    with pytest.raises(ValueError):
        ProviderExtension(namespace, (ProviderExtensionEntry(key, "value"),))


@pytest.mark.parametrize(
    "value",
    ("password=hunter2", "Bearer opaque", "https://private.invalid", "api_key=x"),
)
def test_extensions_reject_sensitive_values_under_innocent_keys(value: str) -> None:
    with pytest.raises(ValueError, match="sensitive data"):
        ProviderExtensionEntry("fixture_note", value)


def test_family_profile_adapter_and_payload_mismatches_fail_closed() -> None:
    local_record, qasm_record, _ = _conformance_records()
    local_case = _case(local_record)
    qasm_case = _case(qasm_record)

    incompatible = run_conformance_case(
        local_case,
        OfflineConformanceDriver(ADAPTER_IDENTITY),
        result_payload=RESULT_PAYLOAD,
    )
    assert incompatible.status is ConformanceStatus.FAILED
    assert incompatible.report is None

    with pytest.raises(ValueError, match="family and target class"):
        ConformanceCase(
            ConformanceTargetFamily.LOCAL_RUNTIME_PLAN,
            qasm_case.target,
            qasm_case.target_ir,
            qasm_case.artifact,
        )
    tampered = replace(qasm_case.artifact, payload=qasm_case.artifact.payload + b"\n")
    with pytest.raises(ValueError, match="does not match"):
        replace(qasm_case, artifact=tampered)


def test_conformance_identity_is_deterministic_across_python_hash_seeds() -> None:
    script = f"""
import json
from pathlib import Path
from tests.internal_ir.test_phase3_provider_conformance import _run
record = json.loads(Path({str(CONFORMANCE_FIXTURES)!r}).read_text())["cases"][2]
print(_run(record).report.conformance_identity)
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


def test_conformance_fixtures_contain_no_real_execution_or_secret_data() -> None:
    text = CONFORMANCE_FIXTURES.read_text(encoding="utf-8").lower()
    forbidden = ("credential", "password", "access_token", "backend_id", "job_id")
    assert all(term not in text for term in forbidden)
    assert "http://" not in text and "https://" not in text


def test_provider_conformance_remains_private() -> None:
    for name in (
        "ConformanceCase",
        "ProviderExtension",
        "run_conformance_case",
    ):
        assert not hasattr(fq, name)
