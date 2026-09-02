from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.capability_comparison import (
    compare_target_capabilities,
)
from flagquantum._compiler.diagnostics import DiagnosticCode
from flagquantum._compiler.target_capabilities import (
    AncillaPolicy,
    ArtifactFormat,
    ArtifactProfile,
    ControlFlowProfile,
    GateCapability,
    MeasurementResult,
    ParameterConstraint,
    TargetCapabilities,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/internal_ir/phase3_batch_a_target_capabilities.json"


def _records() -> list[dict[str, object]]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]


def _target(index: int = 0) -> TargetCapabilities:
    return TargetCapabilities.from_dict(_records()[index]["payload"])


def test_anonymous_fixtures_round_trip_and_match_golden_fingerprints() -> None:
    for record in _records():
        target = TargetCapabilities.from_dict(record["payload"])
        assert TargetCapabilities.from_dict(target.to_dict()) == target
        assert target.semantic_fingerprint == record["expected_fingerprint"]
        assert len(target.semantic_fingerprint) == 64


def test_snapshot_is_immutable_and_isolated_from_caller_mutation() -> None:
    payload = copy.deepcopy(_records()[1]["payload"])
    target = TargetCapabilities.from_dict(payload)
    fingerprint = target.semantic_fingerprint
    payload["native_gates"].append({"operation": "h", "parameters": []})
    payload["topology"]["edges"].append([0, 4])

    assert target.semantic_fingerprint == fingerprint
    with pytest.raises(FrozenInstanceError):
        target.display_label = "changed"


def test_display_label_is_non_semantic_but_capabilities_change_identity() -> None:
    target = _target(1)
    assert replace(
        target, display_label="another anonymous label"
    ).semantic_fingerprint == (target.semantic_fingerprint)
    assert replace(target, supports_noise=True).semantic_fingerprint != (
        target.semantic_fingerprint
    )
    assert replace(target, calibration_snapshot_hash="c" * 64).semantic_fingerprint != (
        target.semantic_fingerprint
    )
    assert (
        replace(
            target,
            artifact_profiles=(ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),),
        ).semantic_fingerprint
        != target.semantic_fingerprint
    )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"target_class": "qasm_text"}, "closed enum"),
        ({"measurement_results": ("counts",)}, "closed enum"),
        ({"supports_reset": 1}, "must be boolean"),
        ({"logical_qubit_capacity": 9}, "logical <= physical"),
        ({"supports_pulse": True}, "requires timing"),
        (
            {"control_flow": ControlFlowProfile.ADAPTIVE_REAL_TIME},
            "requires mid-circuit",
        ),
        ({"maximum_shots": 0}, "must be positive"),
        (
            {
                "ancilla_policy": AncillaPolicy.EXPLICIT_ONLY,
                "maximum_compiler_ancillas": 1,
            },
            "requires zero capacity",
        ),
        ({"calibration_valid_until": "2026-09-02T12:00:00Z"}, "requires a snapshot"),
    ],
)
def test_invalid_combinations_fail_closed(
    update: dict[str, object], message: str
) -> None:
    target = _target()
    with pytest.raises(ValueError, match=message):
        replace(target, **update)


def test_incompatible_artifact_and_unknown_operation_fail_closed() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        replace(
            _target(),
            artifact_profiles=(ArtifactProfile(ArtifactFormat.QCIS_1, "1.0"),),
        )
    with pytest.raises(ValueError, match="unknown native operation"):
        GateCapability("provider_magic_gate")
    with pytest.raises(ValueError, match="unsupported artifact profile version"):
        ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "future")


@pytest.mark.parametrize(
    "forbidden",
    ["provider", "backend_id", "credential", "token", "url", "job_id", "queue"],
)
def test_provider_and_execution_fields_are_rejected(forbidden: str) -> None:
    payload = copy.deepcopy(_records()[0]["payload"])
    payload[forbidden] = "secret-or-mutable-value"
    with pytest.raises(ValueError, match="unknown target capability fields"):
        TargetCapabilities.from_dict(payload)


def test_nested_extension_fields_and_lossy_numeric_coercion_are_rejected() -> None:
    payload = copy.deepcopy(_records()[0]["payload"])
    payload["native_gates"][0]["provider_extension"] = "opaque"
    with pytest.raises(ValueError, match="invalid native gate fields"):
        TargetCapabilities.from_dict(payload)
    with pytest.raises(ValueError, match="qubit capacities must be integers"):
        replace(_target(), logical_qubit_capacity=2.5)
    with pytest.raises(ValueError, match="bounds must be numeric"):
        ParameterConstraint("theta", minimum="-1")
    missing = copy.deepcopy(_records()[0]["payload"])
    del missing["physical_qubit_capacity"]
    with pytest.raises(ValueError, match="missing target capability fields"):
        TargetCapabilities.from_dict(missing)


def test_structured_comparison_reports_all_missing_requirements() -> None:
    available = _target(1)
    required = replace(
        available,
        logical_qubit_capacity=5,
        native_gates=(*available.native_gates, GateCapability("h")),
        measurement_results=(MeasurementResult.COUNTS, MeasurementResult.SAMPLES),
        control_flow=ControlFlowProfile.ADAPTIVE_REAL_TIME,
        supports_mid_circuit_measurement=True,
        ancilla_policy=AncillaPolicy.CLEAN_ALLOCATABLE,
        maximum_compiler_ancillas=1,
    )
    comparison = compare_target_capabilities(required, available)
    fields = {item.field for item in comparison.differences}

    assert comparison.compatible is False
    assert {"native_gates", "measurement_results", "control_flow"} <= fields
    assert {"ancilla_policy", "maximum_compiler_ancillas"} <= fields
    assert all(
        item.code is DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH
        for item in comparison.diagnostics
    )
    assert len(comparison.diagnostics) == len(comparison.differences)


def test_equal_or_stronger_target_is_compatible() -> None:
    target = _target(1)
    assert compare_target_capabilities(target, target).compatible is True


def test_unknown_limit_and_narrower_parameter_domain_fail_closed() -> None:
    required = _target()
    assert not compare_target_capabilities(
        required, replace(required, maximum_shots=None)
    ).compatible
    required_gate = GateCapability("rx")
    narrowed_gate = GateCapability("rx", (ParameterConstraint("theta", -1.0, 1.0),))
    required = replace(required, native_gates=(required_gate,))
    available = replace(required, native_gates=(narrowed_gate,))
    assert not compare_target_capabilities(required, available).compatible


def test_fingerprint_ignores_python_hash_seed() -> None:
    script = f"""
import json
from pathlib import Path
from flagquantum._compiler.target_capabilities import TargetCapabilities
record = json.loads(Path({str(FIXTURES)!r}).read_text())[\"fixtures\"][1]
print(TargetCapabilities.from_dict(record[\"payload\"]).semantic_fingerprint)
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


def test_private_capabilities_do_not_expand_stable_namespace() -> None:
    for name in (
        "TargetCapabilities",
        "TargetClass",
        "ArtifactProfile",
        "compare_target_capabilities",
    ):
        assert not hasattr(fq, name)
