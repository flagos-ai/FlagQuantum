"""Persistence tests for frozen QPU digital-twin models."""

from __future__ import annotations

import json

import pytest

import flagquantum as fq


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {"T1": 40.0, "T2": 60.0, "fidelity": 1.0, "length": 6.4e-8},
            "Q27": {"T1": 35.0, "T2": 50.0, "fidelity": 1.0, "length": 6.4e-8},
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 1.0,
                "length": 2.24e-7,
            }
        },
    }


def _twin():
    return fq.twin.from_quafu_chip_info(
        _chip_info(), target="quafu:Shenglian", qubits=(20, 27)
    )


def test_twin_round_trip_preserves_identity_and_prediction(tmp_path):
    twin = _twin()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    destination = tmp_path / "shenglian-twin.json"

    fq.twin.dump_twin(twin, destination)
    restored = fq.twin.load_twin(destination)

    assert restored.snapshot == twin.snapshot
    assert restored.noise_model.identity == twin.noise_model.identity
    assert restored.predict(circuit) == twin.predict(circuit)
    assert destination.stat().st_mode & 0o777 == 0o600


def test_twin_save_is_idempotent_and_refuses_different_content(tmp_path):
    twin = _twin()
    destination = tmp_path / "twin.json"

    fq.twin.dump_twin(twin, destination)
    fq.twin.dump_twin(twin, destination)

    changed = fq.twin.from_quafu_chip_info(
        {
            **_chip_info(),
            "calibration_time": "2026-08-14 11:30:00",
        },
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    with pytest.raises(ValueError, match="different QPU digital Twin"):
        fq.twin.dump_twin(changed, destination)


def test_twin_load_rejects_unknown_or_identity_mismatched_content(tmp_path):
    twin = _twin()
    valid = tmp_path / "valid.json"
    fq.twin.dump_twin(twin, valid)
    payload = json.loads(valid.read_text(encoding="utf-8"))

    payload["noise_model"]["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical v1 form"):
        fq.twin.load_twin(unknown)

    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["snapshot"]["noise_model_identity"] = "a" * 64
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid QPU digital Twin"):
        fq.twin.load_twin(mismatched)

    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["noise_model"]["rules"] = [False]
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="noise model"):
        fq.twin.load_twin(malformed)


@pytest.mark.parametrize("contents", ["[]", "not json", '{"schema": NaN}'])
def test_twin_load_rejects_invalid_files(tmp_path, contents):
    source = tmp_path / "invalid.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        fq.twin.load_twin(source)


def test_twin_dump_rejects_mutated_model_and_invalid_destination(tmp_path):
    twin = _twin()
    twin.noise_model.device_profile = None

    with pytest.raises(ValueError, match="changed after"):
        fq.twin.dump_twin(twin, tmp_path / "mutated.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid QPU digital Twin"):
        fq.twin.dump_twin(_twin(), invalid)
