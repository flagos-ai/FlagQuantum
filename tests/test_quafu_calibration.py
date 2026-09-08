"""Tests for Quafu calibration conversion."""

import pytest

import flagquantum as fq
from flagquantum.remote import quafu_noise_model_from_chip_info


def _chip_info():
    return {
        "calibration_time": "2026-08-06 19:02:57",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q123": {
                "T1": 36.414,
                "T2": 68.655,
                "fidelity": 0.999,
                "length": 6.4e-8,
            },
            "Q124": {
                "T1": 27.679,
                "T2": 9.455,
                "fidelity": 0.998,
                "length": 6.4e-8,
            },
        },
        "couplers_info": {
            "C145": {
                "qubits_index": [123, 124],
                "fidelity": 0.991,
                "length": 2.24e-7,
            }
        },
    }


def test_quafu_chip_info_builds_timestamped_logical_noise_model():
    matrices = (
        ((0.98, 0.02), (0.09, 0.91)),
        ((0.96, 0.04), (0.12, 0.88)),
    )

    model = quafu_noise_model_from_chip_info(
        _chip_info(), physical_qubits=(123, 124), readout_confusion_matrices=matrices
    )

    assert model.device_profile.captured_at == "2026-08-06 19:02:57+08:00"
    assert model.device_profile.time_unit == "s"
    assert model.device_profile.qubits[0].t1 == pytest.approx(36.414e-6)
    assert model.device_profile.duration_for("ry", (1,)) == pytest.approx(6.4e-8)
    assert model.device_profile.duration_for("u3", (1,)) == pytest.approx(6.4e-8)
    assert any("u3" in rule.gate_names for rule in model.rules)
    assert model.readout_rules[1].error.probabilities == matrices[1]
    instruction = next(iter(fq.Circuit(2).ry(1, 0.2).to_ir()))
    channels = list(model.channels_for(instruction))
    assert channels[0][0].parameters == (("probability", pytest.approx(0.003)),)
    assert channels[0][1] == (1,)

    cz_instruction = next(iter(fq.Circuit(2).cz(0, 1).to_ir()))
    cz_channels = list(model.channels_for(cz_instruction))
    assert len(cz_channels) == 1
    assert cz_channels[0][0].name == "two_qubit_depolarizing"
    assert cz_channels[0][0].parameters == (("probability", pytest.approx(0.01125)),)
    assert cz_channels[0][1] == (0, 1)
    assert model.device_profile.duration_for("cz", (0, 1)) == pytest.approx(2.24e-7)
    assert model.device_profile.duration_for("cx", (0, 1)) == pytest.approx(2.24e-7)


def test_quafu_chip_info_preserves_per_qubit_gate_fidelity():
    model = quafu_noise_model_from_chip_info(_chip_info(), physical_qubits=(123, 124))

    q0 = next(iter(fq.Circuit(2).ry(0, 0.2).to_ir()))
    q1 = next(iter(fq.Circuit(2).ry(1, 0.2).to_ir()))
    p0 = list(model.channels_for(q0))[0][0].parameters[0][1]
    p1 = list(model.channels_for(q1))[0][0].parameters[0][1]

    assert p0 == pytest.approx(0.0015)
    assert p1 == pytest.approx(0.003)


def test_quafu_chip_info_rejects_missing_or_unphysical_calibration():
    payload = _chip_info()
    payload["qubits_info"]["Q123"]["T2"] = 100.0

    with pytest.raises(ValueError, match="invalid T1/T2"):
        quafu_noise_model_from_chip_info(payload, physical_qubits=(123,))
