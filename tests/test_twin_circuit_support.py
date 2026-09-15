"""Topology- and depth-qualified Twin support scenarios."""

from __future__ import annotations

import pytest

import flagquantum as fq


def _chip_info():
    return {
        "calibration_time": "2026-08-14 10:30:00",
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            f"Q{qubit}": {
                "T1": 40.0,
                "T2": 60.0,
                "fidelity": 1.0,
                "length": 6.4e-8,
            }
            for qubit in (20, 27, 34)
        },
        "couplers_info": {
            "C0": {
                "qubits_index": [20, 27],
                "fidelity": 1.0,
                "length": 2.24e-7,
            },
            "C1": {
                "qubits_index": [27, 34],
                "fidelity": 1.0,
                "length": 2.24e-7,
            },
        },
    }


def _fixture():
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(), target="quafu:Shenglian", qubits=(20, 27, 34)
    )
    verified = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=(20, 27, 34),
        supported_operations=("h", "cx", "rx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(verified.to_ir().content_hash,),
        evidence_identity="e" * 64,
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    support = fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((20, 27), (27, 20), (27, 34), (34, 27)),
        maximum_circuit_depth=3,
    )
    return twin, verified, support


def test_verified_circuit_inside_topology_keeps_evidence() -> None:
    twin, verified, support = _fixture()

    report = support.evidence_report(twin, verified)

    assert report.status == "exact_circuit_verified"
    assert report.tv_error_bound == pytest.approx(0.04)
    assert support.supports(verified)


def test_supported_report_predicts_full_computational_basis_distribution() -> None:
    twin, verified, support = _fixture()

    report = support.evidence_report(twin, verified)

    assert report.prediction is not None
    assert report.prediction.ideal_probabilities == pytest.approx(
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    )
    assert len(report.prediction.twin_probabilities) == 2**verified.n_qubits


def test_unverified_physical_coupler_fails_closed() -> None:
    twin, _, support = _fixture()
    circuit = fq.Circuit(3).cx(0, 2)

    report = support.evidence_report(twin, circuit)

    assert report.status == "out_of_scope"
    assert report.tv_error_bound is None
    assert report.prediction is None
    assert "physical_coupler_outside_support" in report.reasons


def test_unverified_reverse_direction_fails_closed() -> None:
    twin, _, support = _fixture()
    directional = fq.twin.TwinCircuitSupport(
        evidence=support.evidence,
        directed_couplers=((20, 27), (27, 34)),
        maximum_circuit_depth=3,
    )
    circuit = fq.Circuit(3).cx(1, 0)

    report = directional.evidence_report(twin, circuit)

    assert report.status == "out_of_scope"
    assert report.tv_error_bound is None
    assert "physical_coupler_outside_support" in report.reasons


def test_circuit_deeper_than_evidence_fails_closed() -> None:
    twin, _, support = _fixture()
    circuit = fq.Circuit(3).rx(0, 0.1).rx(0, 0.2).rx(0, 0.3).rx(0, 0.4)

    report = support.evidence_report(twin, circuit)

    assert report.status == "out_of_scope"
    assert report.tv_error_bound is None
    assert "maximum_circuit_depth_exceeded" in report.reasons


def test_wider_circuit_fails_closed_before_physical_mapping() -> None:
    twin, _, support = _fixture()
    circuit = fq.Circuit(4).cx(2, 3)

    report = support.evidence_report(twin, circuit)

    assert report.status == "out_of_scope"
    assert report.prediction is None
    assert "outside_evidence_structure" in report.reasons


def test_support_rejects_invalid_physical_couplers() -> None:
    _, _, support = _fixture()

    with pytest.raises(ValueError, match="evidence mapping"):
        fq.twin.TwinCircuitSupport(
            evidence=support.evidence,
            directed_couplers=((20, 99),),
            maximum_circuit_depth=3,
        )

    with pytest.raises(ValueError, match="exactly two"):
        fq.twin.TwinCircuitSupport(
            evidence=support.evidence,
            directed_couplers=((20,),),
            maximum_circuit_depth=3,
        )


def test_support_round_trip_is_deterministic(tmp_path) -> None:
    _, _, support = _fixture()
    destination = tmp_path / "twin-circuit-support.json"

    fq.twin.dump_circuit_support(support, destination)
    restored = fq.twin.load_circuit_support(destination)
    fq.twin.dump_circuit_support(restored, destination)

    assert restored == support
    assert restored.identity == support.identity
    assert destination.stat().st_mode & 0o777 == 0o600


def test_support_refuses_to_replace_different_artifact(tmp_path) -> None:
    _, _, support = _fixture()
    destination = tmp_path / "twin-circuit-support.json"
    fq.twin.dump_circuit_support(support, destination)
    different = fq.twin.TwinCircuitSupport(
        evidence=support.evidence,
        directed_couplers=support.directed_couplers,
        maximum_circuit_depth=2,
    )

    with pytest.raises(ValueError, match="Refusing to replace different"):
        fq.twin.dump_circuit_support(different, destination)
