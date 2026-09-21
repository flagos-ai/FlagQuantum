"""Connected Twin-region model composition scenarios."""

from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.noise import NoiseModel, bit_flip_channel

pytestmark = pytest.mark.integration


def _chip_info(
    *,
    overlap_t1: float = 40.0,
    overlap_fidelity: float = 0.998,
):
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            "Q20": {
                "T1": 41.0,
                "T2": 61.0,
                "fidelity": 0.997,
                "length": 6.4e-8,
            },
            "Q27": {
                "T1": overlap_t1,
                "T2": 60.0,
                "fidelity": overlap_fidelity,
                "length": 6.4e-8,
            },
            "Q34": {
                "T1": 42.0,
                "T2": 62.0,
                "fidelity": 0.996,
                "length": 6.4e-8,
            },
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 0.985,
                "length": 2.24e-7,
            },
            "C1": {
                "qubits_index": [27, 34],
                "fidelity": 0.984,
                "length": 2.24e-7,
            },
        },
    }


def _support(twin: fq.twin.QPUDigitalTwin) -> fq.twin.TwinCircuitSupport:
    qubits = twin.snapshot.physical_qubits
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=qubits,
        supported_operations=("h", "cx", "rx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(circuit.to_ir().content_hash,),
        evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    return fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=4,
    )


def _cell(
    qubits: tuple[int, int],
    *,
    chip_info=None,
):
    twin = fq.twin.from_quafu_chip_info(
        _chip_info() if chip_info is None else chip_info,
        target="quafu:Shenglian",
        qubits=qubits,
    )
    return twin, _support(twin)


def _region_model() -> fq.twin.TwinRegionModel:
    return fq.twin.compose_region_twin([_cell((20, 27)), _cell((27, 34))])


def test_region_twin_predicts_one_connected_three_qubit_circuit() -> None:
    model = _region_model()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)

    prediction = model.predict(
        circuit,
        physical_qubits=(20, 27, 34),
    )

    assert model.target == "quafu:Shenglian"
    assert model.physical_qubits == (20, 27, 34)
    assert model.twin.snapshot.physical_qubits == (20, 27, 34)
    assert prediction.n_wires == 3
    assert len(prediction.twin_probabilities) == 8
    assert sum(prediction.twin_probabilities) == pytest.approx(1.0)
    assert prediction.total_variation_from_ideal > 0


def test_region_twin_is_deterministic() -> None:
    first = _region_model()
    second = _region_model()

    assert first.identity == second.identity
    assert first.region.identity == second.region.identity
    assert first.twin.snapshot.identity == second.twin.snapshot.identity


def test_region_twin_requires_its_exact_regional_wire_order() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        _region_model().predict(
            fq.Circuit(3).h(0),
            physical_qubits=(27, 20, 34),
        )


def test_region_twin_rejects_a_circuit_outside_structural_scope() -> None:
    with pytest.raises(ValueError, match="physical_couplers_outside_region"):
        _region_model().predict(
            fq.Circuit(3).cx(2, 0),
            physical_qubits=(20, 27, 34),
        )


def test_region_twin_rejects_conflicting_overlap_calibration() -> None:
    with pytest.raises(ValueError, match="disagree on qubit calibration"):
        fq.twin.compose_region_twin(
            [
                _cell((20, 27)),
                _cell((27, 34), chip_info=_chip_info(overlap_t1=39.0)),
            ]
        )


def test_region_twin_rejects_conflicting_overlap_noise_channel() -> None:
    first, first_support = _cell((20, 27))
    second, _ = _cell((27, 34))
    changed = NoiseModel.from_dict(second.noise_model.to_dict())
    changed.add("h", bit_flip_channel(0.1), wires=(0,))
    changed_twin = fq.twin.from_noise_model(
        changed,
        target="quafu:Shenglian",
        qubits=(27, 34),
    )
    changed_support = _support(changed_twin)

    with pytest.raises(ValueError, match="scoped noise channel"):
        fq.twin.compose_region_twin(
            [(first, first_support), (changed_twin, changed_support)]
        )


def test_region_twin_rejects_cell_local_unscoped_noise() -> None:
    first, first_support = _cell((20, 27))
    changed = NoiseModel.from_dict(first.noise_model.to_dict())
    changed.add("h", bit_flip_channel(0.01))
    changed_first = fq.twin.from_noise_model(
        changed,
        target="quafu:Shenglian",
        qubits=(20, 27),
    )
    changed_first_support = _support(changed_first)

    with pytest.raises(ValueError, match="identical in every"):
        fq.twin.compose_region_twin(
            [(changed_first, changed_first_support), _cell((27, 34))]
        )


def test_region_twin_exposes_no_region_accuracy_report() -> None:
    model = _region_model()

    assert not hasattr(model, "evidence_report")
    assert not hasattr(model, "tv_error_bound")
    assert not hasattr(model, "confidence_level")
