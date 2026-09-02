from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flagquantum._compiler.exporters.circuit_ir import (
    ExportStatus,
    export_circuit_ir,
    seal_circuit_ir_round_trip,
)
from flagquantum._compiler.import_models import ImportStatus
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/internal_ir/circuit_ir_v1/manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


SUPPORTED_FIXTURES = [
    fixture
    for fixture in _manifest()["fixtures"]
    if fixture["expected_import"] == "supported_exact"
]


@pytest.mark.parametrize(
    "fixture",
    SUPPORTED_FIXTURES,
    ids=lambda fixture: fixture["id"],
)
def test_supported_fixture_round_trip_is_canonical_payload_exact(fixture) -> None:
    source = CircuitIR.from_dict(fixture["payload"])

    sealed = seal_circuit_ir_round_trip(source)
    exported = export_circuit_ir(sealed.artifact)

    assert sealed.status is ImportStatus.SUPPORTED_EXACT
    assert sealed.ok is True
    assert exported.status is ExportStatus.EXPORTED_EXACT
    assert exported.ok is True
    assert exported.diagnostics == ()
    assert exported.circuit_ir.to_dict() == source.to_dict()
    assert exported.circuit_ir.to_json() == source.to_json()
    assert exported.circuit_ir.content_hash == source.content_hash


def test_unsupported_source_never_produces_a_round_trip_artifact() -> None:
    dynamic = next(
        fixture
        for fixture in _manifest()["fixtures"]
        if fixture["expected_import"] == "unsupported_with_diagnostics"
    )
    source = CircuitIR.from_dict(dynamic["payload"])

    sealed = seal_circuit_ir_round_trip(source)

    assert sealed.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert sealed.artifact is None
    assert sealed.diagnostics


def test_non_circuit_source_and_non_artifact_export_fail_closed() -> None:
    sealed = seal_circuit_ir_round_trip(object())
    exported = export_circuit_ir(object())

    assert sealed.status is ImportStatus.INVALID_INPUT
    assert sealed.artifact is None
    assert exported.status is ExportStatus.INVALID_ARTIFACT
    assert exported.circuit_ir is None
    assert exported.diagnostics


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("canonical_source_json", "{}", "payload hash mismatch"),
        ("source_content_hash", "0" * 64, "source identity"),
        ("internal_program_identity", "0" * 64, "program identity"),
        ("profile", "provider_codegen", "unsupported round-trip profile"),
    ],
)
def test_tampered_artifact_is_rejected_without_lossy_recovery(
    field: str, value: str, message: str
) -> None:
    source = CircuitIR.from_dict(SUPPORTED_FIXTURES[0]["payload"])
    artifact = seal_circuit_ir_round_trip(source).artifact
    tampered = replace(artifact, **{field: value})

    result = export_circuit_ir(tampered)

    assert result.status is ExportStatus.INVALID_ARTIFACT
    assert result.circuit_ir is None
    assert message in result.diagnostics[0].message
    assert any(
        "never performs lossy recovery" in note for note in result.diagnostics[0].notes
    )


def test_exporter_has_no_best_effort_or_provider_codegen_parameter() -> None:
    parameters = inspect.signature(export_circuit_ir).parameters

    assert tuple(parameters) == ("artifact",)
    assert "best_effort" not in parameters
    assert "provider" not in parameters
