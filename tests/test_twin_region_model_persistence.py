"""Regional Twin model persistence scenarios and identity tamper tests."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import flagquantum as fq
from tests.test_twin_region_candidate_holdout import (
    _evaluate,
    _holdout_circuits,
    _region,
    _study_for,
    _with_predictions,
)

pytestmark = pytest.mark.integration

_REGION_MAPPING = (20, 27, 34)
_INCUMBENT_CAPTURE = "2026-08-14 10:30:00"
_CANDIDATE_CAPTURE = "2026-08-15 10:30:00"


def _released():
    incumbent = _region(_INCUMBENT_CAPTURE, 0.96)
    candidate = _region(_CANDIDATE_CAPTURE, 0.99)
    study = _with_predictions(_study_for(incumbent, candidate))
    evaluation = _evaluate(study)
    release = fq.twin.release_region_candidate(
        incumbent,
        candidate,
        study=study,
        evaluation=evaluation,
    )
    return candidate, release


def _stored_region_twin(tmp_path: Path, region_twin: fq.twin.TwinRegionModel):
    destination = tmp_path / "region-twin.json"
    fq.twin.dump_region_twin(region_twin, destination)
    return destination


def _payload(destination: Path) -> dict:
    return json.loads(destination.read_text(encoding="utf-8"))


def _rewrite(destination: Path, payload: dict) -> None:
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_application_restores_a_released_regional_twin_in_a_new_process(
    tmp_path,
) -> None:
    candidate, release = _released()
    circuit = _holdout_circuits()[0]

    _stored_region_twin(tmp_path, candidate)
    fq.twin.dump_region_release(release, tmp_path / "region-release.json")
    restored = fq.twin.load_region_twin(tmp_path / "region-twin.json")
    restored_release = fq.twin.load_region_release(tmp_path / "region-release.json")

    assert restored.identity == candidate.identity
    assert restored.target == "quafu:Shenglian"
    assert restored.physical_qubits == _REGION_MAPPING

    assessment = restored_release.assess(
        restored,
        circuit,
        physical_qubits=_REGION_MAPPING,
    )

    assert assessment.status == "released_exact_circuit"
    assert assessment.prediction is not None
    assert assessment.prediction.twin_probabilities == pytest.approx(
        candidate.predict(circuit, physical_qubits=_REGION_MAPPING).twin_probabilities
    )


def test_round_trip_preserves_identity_and_prediction_parity(tmp_path) -> None:
    candidate, _ = _released()
    circuit = _holdout_circuits()[1]

    restored = fq.twin.load_region_twin(_stored_region_twin(tmp_path, candidate))

    assert restored.identity == candidate.identity
    assert restored.region == candidate.region
    assert restored.twin.snapshot == candidate.twin.snapshot
    assert restored.twin.noise_model.identity == candidate.twin.noise_model.identity
    expected = candidate.predict(circuit, physical_qubits=_REGION_MAPPING)
    actual = restored.predict(circuit, physical_qubits=_REGION_MAPPING)
    assert actual == expected
    assert actual.twin_probabilities == pytest.approx(expected.twin_probabilities)
    assert actual.total_variation_from_ideal == pytest.approx(
        expected.total_variation_from_ideal
    )


def test_artifact_is_a_private_canonical_model_without_operational_state(
    tmp_path,
) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert payload["schema"] == "flagquantum.twin_region_model_artifact.v1"
    assert set(payload) == {"schema", "region", "twin"}
    assert payload["region"]["schema"] == "flagquantum.twin_connected_region.v1"
    assert payload["twin"]["snapshot"]["schema"] == "flagquantum.twin_snapshot.v1"
    assert payload["twin"]["noise_model"]["schema"] == "flagquantum.noise_model.v1"
    assert (
        destination.read_text(encoding="utf-8")
        == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    encoded = destination.read_text(encoding="utf-8").lower()
    for forbidden in (
        "credential",
        "api_key",
        "token",
        "task_id",
        "routing_authorized",
        "release_decision",
        "confidence_level",
        "tv_error_bound",
    ):
        assert forbidden not in encoded


def test_repeated_identical_write_is_a_noop(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    before = destination.read_bytes()

    fq.twin.dump_region_twin(candidate, destination)

    assert destination.read_bytes() == before


def test_dump_refuses_to_replace_a_different_region_twin(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    before = destination.read_bytes()
    other = _region(_INCUMBENT_CAPTURE, 0.96)

    with pytest.raises(ValueError, match="Refusing to replace different"):
        fq.twin.dump_region_twin(other, destination)

    assert destination.read_bytes() == before


def test_dump_refuses_to_replace_an_invalid_artifact(tmp_path) -> None:
    candidate, _ = _released()
    destination = tmp_path / "region-twin.json"
    destination.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to replace invalid"):
        fq.twin.dump_region_twin(candidate, destination)

    assert destination.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("value", [17, [], "region-twin", None])
def test_dump_requires_a_region_model(tmp_path, value) -> None:
    with pytest.raises(TypeError, match="region_twin must be a TwinRegionModel"):
        fq.twin.dump_region_twin(value, tmp_path / "region-twin.json")


def test_load_rejects_unknown_schema_and_non_object_files(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["schema"] = "flagquantum.twin_region_model_artifact.v2"
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="unsupported Twin region-model artifact"):
        fq.twin.load_region_twin(destination)

    listing = tmp_path / "list.json"
    listing.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        fq.twin.load_region_twin(listing)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Cannot load Twin region model"):
        fq.twin.load_region_twin(missing)


@pytest.mark.parametrize("field", ["schema", "region", "twin"])
def test_load_rejects_missing_artifact_fields(tmp_path, field) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    del payload[field]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="fields do not match the v1 schema"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_extra_artifact_fields(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["routing_authorized"] = True
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="unexpected=\\['routing_authorized'\\]"):
        fq.twin.load_region_twin(destination)


@pytest.mark.parametrize("member", ["region", "twin"])
def test_load_rejects_non_object_members_and_extra_member_fields(
    tmp_path, member
) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload[member] = "flagquantum"
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="members must be JSON objects"):
        fq.twin.load_region_twin(destination)

    destination.unlink()
    _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload[member]["undeclared"] = 1
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="fields do not match the v1 schema"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_a_retargeted_region(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["backend_name"] = "OtherChip"
    _rewrite(destination, payload)

    # A canonical region that no longer matches the frozen snapshot is an
    # identity change, not a variant of the stored model.
    with pytest.raises(ValueError, match="Invalid Twin region model"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_a_changed_capture_time(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["captured_at"] = "2026-08-16T10:30:00+08:00"
    payload["twin"]["snapshot"]["captured_at"] = "2026-08-16T10:30:00+08:00"
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_reordered_physical_mapping(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["physical_qubits"] = [34, 27, 20]
    payload["twin"]["snapshot"]["physical_qubits"] = [34, 27, 20]
    _rewrite(destination, payload)

    # The composed calibration is bound to the region it was composed for, so a
    # silently reordered mapping cannot be reloaded as an equivalent model.
    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_a_redirected_coupler(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    couplers = payload["region"]["directed_couplers"]
    payload["region"]["directed_couplers"] = [
        [couplers[0][1], couplers[0][0]],
        *couplers[1:],
    ]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_tampered_source_or_support_identities(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["source_snapshot_identities"] = ["0" * 64, "1" * 64]
    _rewrite(destination, payload)

    # A rewritten source identity is still canonical JSON, so only the identity
    # binding between the region and its composed calibration can refuse it.
    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.load_region_twin(destination)

    destination.unlink()
    _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["source_support_identities"] = ["0" * 64, "1" * 64]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_a_tampered_noise_model(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["twin"]["noise_model"]["rules"][0]["channel"]["probabilities"] = [0.5, 0.5]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="noise model is not in canonical v1 form"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_a_dropped_region_scope_field(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    del payload["twin"]["snapshot"]["noise_model_identity"]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="Invalid Twin region-model twin"):
        fq.twin.load_region_twin(destination)


def test_load_rejects_noncanonical_but_equivalent_content(tmp_path) -> None:
    candidate, _ = _released()
    destination = _stored_region_twin(tmp_path, candidate)
    payload = _payload(destination)
    payload["region"]["physical_qubits"] = [
        float(qubit) for qubit in payload["region"]["physical_qubits"]
    ]
    _rewrite(destination, payload)

    with pytest.raises(ValueError, match="canonical v1 form"):
        fq.twin.load_region_twin(destination)


def test_dump_rejects_a_mutated_device_profile(tmp_path) -> None:
    candidate, _ = _released()
    profile = candidate.twin.noise_model.device_profile
    assert profile is not None
    object.__setattr__(profile, "source", "flagquantum:twin-region:tampered")
    # Rewriting the source also changes the calibration identity, so the frozen
    # snapshot refuses the model before the region binding is even consulted.
    assert profile.identity != candidate.twin.snapshot.calibration_identity

    with pytest.raises(ValueError, match="changed after it was frozen"):
        fq.twin.dump_region_twin(candidate, tmp_path / "region-twin.json")


def test_dump_rejects_an_identity_changing_model(tmp_path) -> None:
    candidate, _ = _released()
    tampered = replace(
        candidate,
        region=replace(candidate.region, maximum_circuit_depth=1),
    )

    # ``replace`` bypasses the region invariants, so the writer must refuse a
    # model whose composed calibration no longer belongs to its region.
    with pytest.raises(ValueError, match="does not match its region"):
        fq.twin.dump_region_twin(tampered, tmp_path / "region-twin.json")


def test_writer_accepts_an_explicit_destination_path(tmp_path) -> None:
    candidate, _ = _released()
    destination = os.fspath(tmp_path / "region-twin.json")

    fq.twin.dump_region_twin(candidate, destination)

    assert fq.twin.load_region_twin(destination).identity == candidate.identity
