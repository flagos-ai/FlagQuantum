"""Connected Twin-region model composition scenarios."""

from __future__ import annotations

import hashlib

import pytest

import flagquantum as fq
from examples.remote import quafu_twin_region_validation
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.remote.qpu import (
    DeploymentResult,
    ProviderTaskHandle,
    build_result_metadata,
)

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


def _hardware_report(
    experiment: fq.twin.TwinExperiment,
    *,
    task_id: str,
    counts: dict[str, int],
) -> fq.twin.TwinHardwareReport:
    digest = hashlib.sha256(experiment.submitted_qasm.encode()).hexdigest()
    receipt = ProviderTaskHandle(
        provider="quafu",
        task_id=task_id,
        backend_name=experiment.backend_name,
        payload={
            "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
            "deployment_package_schema": "flagquantum_submitted_qasm_v1",
            "deployment_program_format": "openqasm-2",
            "routing_evidence_sha256": digest,
            "deployment_artifact_sha256": digest,
            "submitted_qasm_sha256": digest,
            "compiler": None,
            "target_qubits": list(experiment.target_qubits),
        },
    )
    result = DeploymentResult(
        handle=receipt,
        counts=counts,
        shots=sum(counts.values()),
        metadata=build_result_metadata(
            receipt,
            {"transpiled": experiment.submitted_qasm},
        ),
    )
    return experiment.validate_result(result, receipt=receipt)


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


def test_region_twin_prepares_identity_bound_experiment_without_submission() -> None:
    model = _region_model()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)

    experiment = model.prepare_experiment(
        circuit,
        physical_qubits=(20, 27, 34),
        name="regional-ghz",
        shots=1024,
    )

    assert experiment.snapshot_identity == model.twin.snapshot.identity
    assert experiment.target_qubits == model.physical_qubits
    assert experiment.prediction == model.predict(
        circuit,
        physical_qubits=model.physical_qubits,
    )
    assert experiment.submitted_qasm.startswith("OPENQASM 2.0;")


def test_region_twin_builds_exact_support_from_repeated_hardware_results() -> None:
    model = _region_model()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
    experiment = model.prepare_experiment(
        circuit,
        physical_qubits=model.physical_qubits,
        name="regional-ghz",
        shots=1024,
    )
    first = _hardware_report(
        experiment,
        task_id="regional-1",
        counts={"000": 500, "111": 524},
    )
    second = _hardware_report(
        experiment,
        task_id="regional-2",
        counts={"000": 490, "001": 10, "110": 10, "111": 514},
    )
    series = experiment.validation_series((first, second), circuit=circuit)

    support = model.support_from_validation_series(
        series,
        circuit,
        physical_qubits=model.physical_qubits,
    )
    report = support.evidence_report(model.twin, circuit)

    assert support.directed_couplers == ((20, 27), (27, 34))
    assert support.maximum_circuit_depth == 3
    assert report.status == "exact_circuit_verified"
    assert report.tv_error_bound == series.verified_tv_error_bound
    assert report.confidence_level == series.confidence_level


def test_region_twin_requires_repeatability_before_qualifying_support() -> None:
    model = _region_model()
    circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
    experiment = model.prepare_experiment(
        circuit,
        physical_qubits=model.physical_qubits,
        name="regional-ghz",
        shots=1024,
    )
    report = _hardware_report(
        experiment,
        task_id="regional-1",
        counts={"000": 512, "111": 512},
    )
    series = experiment.validation_series((report,), circuit=circuit)

    with pytest.raises(ValueError, match="at least two distinct"):
        model.support_from_validation_series(
            series,
            circuit,
            physical_qubits=model.physical_qubits,
        )


def test_complete_region_validation_example_executes_offline(
    monkeypatch,
    tmp_path,
) -> None:
    twin_a, support_a = _cell((20, 27))
    twin_b, support_b = _cell((27, 34))
    cell_a_twin = tmp_path / "cell-a-twin.json"
    cell_a_support = tmp_path / "cell-a-support.json"
    cell_b_twin = tmp_path / "cell-b-twin.json"
    cell_b_support = tmp_path / "cell-b-support.json"
    validation_path = tmp_path / "regional-validation.json"
    support_path = tmp_path / "regional-support.json"
    fq.twin.dump_twin(twin_a, cell_a_twin)
    fq.twin.dump_circuit_support(support_a, cell_a_support)
    fq.twin.dump_twin(twin_b, cell_b_twin)
    fq.twin.dump_circuit_support(support_b, cell_b_support)

    class Provider:
        provider = "quafu"

        def __init__(self) -> None:
            self.submissions = 0
            self.programs: dict[str, str] = {}

        def submit_qasm(self, qasm, *, chip, name, shots, target_qubits):
            assert (chip, name, shots, target_qubits) == (
                "Shenglian",
                "flagquantum-regional-ghz",
                1024,
                (20, 27, 34),
            )
            self.submissions += 1
            task_id = f"regional-example-{self.submissions}"
            self.programs[task_id] = qasm
            digest = hashlib.sha256(qasm.encode()).hexdigest()
            return ProviderTaskHandle(
                provider="quafu",
                task_id=task_id,
                backend_name=chip,
                payload={
                    "deployment_receipt_schema": "flagquantum_submission_receipt_v1",
                    "deployment_package_schema": "flagquantum_submitted_qasm_v1",
                    "deployment_program_format": "openqasm-2",
                    "routing_evidence_sha256": digest,
                    "deployment_artifact_sha256": digest,
                    "submitted_qasm_sha256": digest,
                    "compiler": None,
                    "target_qubits": list(target_qubits),
                },
            )

        def query_status(self, receipt):
            assert receipt.task_id in self.programs
            return "Finished"

        def fetch_result(self, receipt):
            return DeploymentResult(
                handle=receipt,
                counts={"000": 512, "111": 512},
                shots=1024,
                metadata=build_result_metadata(
                    receipt,
                    {"transpiled": self.programs[receipt.task_id]},
                ),
            )

    provider = Provider()
    monkeypatch.setenv("QUAFU_API_TOKEN", "offline-test-token")
    monkeypatch.setattr(quafu_twin_region_validation, "QuafuProvider", lambda: provider)
    monkeypatch.setattr(quafu_twin_region_validation, "CELL_A_TWIN", cell_a_twin)
    monkeypatch.setattr(quafu_twin_region_validation, "CELL_A_SUPPORT", cell_a_support)
    monkeypatch.setattr(quafu_twin_region_validation, "CELL_B_TWIN", cell_b_twin)
    monkeypatch.setattr(quafu_twin_region_validation, "CELL_B_SUPPORT", cell_b_support)
    monkeypatch.setattr(
        quafu_twin_region_validation,
        "VALIDATION_PATH",
        validation_path,
    )
    monkeypatch.setattr(quafu_twin_region_validation, "SUPPORT_PATH", support_path)

    quafu_twin_region_validation.main()

    assert provider.submissions == 2
    assert fq.twin.load_validation_series(validation_path).repetitions == 2
    assert fq.twin.load_circuit_support(support_path).evidence.physical_qubits == (
        20,
        27,
        34,
    )
