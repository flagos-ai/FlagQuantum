"""Connected Twin-region structural coverage scenarios."""

from __future__ import annotations

from dataclasses import replace

import pytest

import flagquantum as fq


def _chip_info(*, captured_at: str = "2026-08-14 10:30:00"):
    return {
        "calibration_time": captured_at,
        "basis_gates": ["h", "rx", "ry", "rz", "cz"],
        "qubits_info": {
            f"Q{qubit}": {
                "T1": 40.0,
                "T2": 60.0,
                "fidelity": 1.0,
                "length": 6.4e-8,
            }
            for qubit in (20, 27, 34, 41)
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
            "C2": {
                "qubits_index": [34, 41],
                "fidelity": 1.0,
                "length": 2.24e-7,
            },
        },
    }


def _cell(
    qubits: tuple[int, int],
    *,
    target: str = "quafu:Shenglian",
    captured_at: str = "2026-08-14 10:30:00",
):
    twin = fq.twin.from_quafu_chip_info(
        _chip_info(captured_at=captured_at),
        target=target,
        qubits=qubits,
    )
    verified = fq.Circuit(2).h(0).cx(0, 1)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=qubits,
        supported_operations=("h", "cx", "rx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(verified.to_ir().content_hash,),
        evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    support = fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=4,
    )
    return twin, support


def _region():
    return fq.twin.compose_connected_region([_cell((20, 27)), _cell((27, 34))])


def test_connected_cells_cover_an_explicit_three_qubit_mapping() -> None:
    region = _region()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)

    report = region.coverage_report(
        circuit,
        physical_qubits=(20, 27, 34),
    )

    assert region.target == "quafu:Shenglian"
    assert region.physical_qubits == (20, 27, 34)
    assert report.status == "covered"
    assert report.covered_directed_couplers == ((20, 27), (27, 34))
    assert report.missing_qubits == ()
    assert report.missing_directed_couplers == ()
    assert report.tv_error_bound is None
    assert report.confidence_level is None


def test_region_does_not_infer_a_reverse_coupler() -> None:
    region = _region()
    circuit = fq.Circuit(3).cx(1, 0)

    report = region.coverage_report(circuit, physical_qubits=(20, 27, 34))

    assert report.status == "out_of_scope"
    assert report.missing_directed_couplers == ((27, 20),)
    assert "physical_couplers_outside_region" in report.reasons
    assert report.tv_error_bound is None


def test_region_reports_missing_qubits_without_inventing_coverage() -> None:
    region = _region()
    circuit = fq.Circuit(3).h(0).cx(1, 2)

    report = region.coverage_report(circuit, physical_qubits=(20, 27, 41))

    assert report.status == "out_of_scope"
    assert report.covered_qubits == (20, 27)
    assert report.missing_qubits == (41,)
    assert report.missing_directed_couplers == ((27, 41),)
    assert report.tv_error_bound is None


def test_region_uses_the_most_conservative_cell_limits() -> None:
    first_twin, first = _cell((20, 27))
    second_twin, second = _cell((27, 34))
    second = fq.twin.TwinCircuitSupport(
        evidence=second.evidence,
        directed_couplers=second.directed_couplers,
        maximum_circuit_depth=1,
    )
    region = fq.twin.compose_connected_region(
        [(first_twin, first), (second_twin, second)]
    )

    report = region.coverage_report(
        fq.Circuit(3).h(0).cx(0, 1),
        physical_qubits=(20, 27, 34),
    )

    assert region.maximum_circuit_depth == 1
    assert report.status == "out_of_scope"
    assert "maximum_circuit_depth_exceeded" in report.reasons


def test_region_rejects_cells_from_different_targets() -> None:
    with pytest.raises(ValueError, match="same provider and backend"):
        fq.twin.compose_connected_region(
            [_cell((20, 27)), _cell((27, 34), target="quafu:Baihua")]
        )


def test_region_rejects_cells_from_different_capture_times() -> None:
    with pytest.raises(ValueError, match="calibration capture time"):
        fq.twin.compose_connected_region(
            [
                _cell((20, 27)),
                _cell((27, 34), captured_at="2026-08-14 11:30:00"),
            ]
        )


def test_region_rejects_disconnected_cells() -> None:
    with pytest.raises(ValueError, match="one connected region"):
        fq.twin.compose_connected_region([_cell((20, 27)), _cell((34, 41))])


def test_region_requires_a_complete_explicit_mapping() -> None:
    with pytest.raises(ValueError, match="map every logical"):
        _region().coverage_report(
            fq.Circuit(3).h(0),
            physical_qubits=(20, 27),
        )


def test_region_identity_and_report_are_deterministic() -> None:
    first = _region()
    second = _region()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)

    assert first.identity == second.identity
    assert first.to_dict() == second.to_dict()
    assert (
        first.coverage_report(circuit, physical_qubits=(20, 27, 34)).to_dict()
        == second.coverage_report(circuit, physical_qubits=(20, 27, 34)).to_dict()
    )


def test_region_coverage_rejects_injected_statistical_claims() -> None:
    report = _region().coverage_report(
        fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
        physical_qubits=(20, 27, 34),
    )

    with pytest.raises(ValueError, match="cannot carry statistical bounds"):
        replace(report, tv_error_bound=0.01)
