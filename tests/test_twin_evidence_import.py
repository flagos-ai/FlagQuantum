"""Offline import scenarios for canonical QPU Twin evidence."""

from __future__ import annotations

import json

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration


def _payload() -> dict[str, object]:
    return {
        "schema": "flagquantum.twin_evidence_envelope.v1",
        "snapshot_identity": "a" * 64,
        "physical_qubits": [82, 75, 68, 62],
        "supported_operations": ["h", "ry", "cz", "measure"],
        "maximum_instruction_count": 16,
        "verified_circuit_identities": ["b" * 64, "c" * 64],
        "evidence_identity": "d" * 64,
        "verified_tv_error_bound": 0.14,
        "estimated_tv_error_bound": None,
        "confidence_level": 0.95,
    }


def test_load_evidence_round_trips_canonical_envelope(tmp_path):
    source = tmp_path / "twin-evidence.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    evidence = fq.twin.load_evidence(source)

    assert evidence.to_dict() == _payload()
    assert (
        evidence.identity == fq.twin.TwinEvidenceEnvelope.from_dict(_payload()).identity
    )


def test_dump_evidence_writes_canonical_round_trip(tmp_path):
    destination = tmp_path / "twin-evidence.json"
    evidence = fq.twin.TwinEvidenceEnvelope.from_dict(_payload())

    fq.twin.dump_evidence(evidence, destination)

    assert destination.read_text(encoding="utf-8") == (
        json.dumps(_payload(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert fq.twin.load_evidence(destination) == evidence


def test_dump_evidence_is_idempotent_for_the_same_identity(tmp_path):
    destination = tmp_path / "twin-evidence.json"
    evidence = fq.twin.TwinEvidenceEnvelope.from_dict(_payload())
    fq.twin.dump_evidence(evidence, destination)
    original = destination.read_bytes()

    fq.twin.dump_evidence(evidence, destination)

    assert destination.read_bytes() == original


def test_dump_evidence_refuses_to_replace_different_evidence(tmp_path):
    destination = tmp_path / "twin-evidence.json"
    first = fq.twin.TwinEvidenceEnvelope.from_dict(_payload())
    changed_payload = _payload()
    changed_payload["evidence_identity"] = "e" * 64
    changed = fq.twin.TwinEvidenceEnvelope.from_dict(changed_payload)
    fq.twin.dump_evidence(first, destination)

    with pytest.raises(ValueError, match="Refusing to replace different"):
        fq.twin.dump_evidence(changed, destination)

    assert fq.twin.load_evidence(destination) == first


def test_dump_evidence_refuses_to_replace_invalid_file(tmp_path):
    destination = tmp_path / "twin-evidence.json"
    destination.write_text("not evidence", encoding="utf-8")
    evidence = fq.twin.TwinEvidenceEnvelope.from_dict(_payload())

    with pytest.raises(ValueError, match="Refusing to replace invalid"):
        fq.twin.dump_evidence(evidence, destination)

    assert destination.read_text(encoding="utf-8") == "not evidence"


def test_dump_evidence_requires_an_envelope(tmp_path):
    with pytest.raises(TypeError, match="TwinEvidenceEnvelope"):
        fq.twin.dump_evidence(_payload(), tmp_path / "evidence.json")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("evidence_identity"),
        lambda payload: payload.update({"research_campaign": "q-atlas"}),
        lambda payload: payload.update({"schema": "unknown"}),
        lambda payload: payload.update({"snapshot_identity": "not-a-digest"}),
        lambda payload: payload.update({"confidence_level": None}),
    ],
)
def test_load_evidence_rejects_noncanonical_or_unbounded_payloads(tmp_path, mutate):
    payload = _payload()
    mutate(payload)
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        fq.twin.load_evidence(source)


@pytest.mark.parametrize("contents", ["[]", "{", "null"])
def test_load_evidence_rejects_non_object_json(tmp_path, contents):
    source = tmp_path / "invalid.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        fq.twin.load_evidence(source)
