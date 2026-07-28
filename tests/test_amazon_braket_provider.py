from types import SimpleNamespace

import pytest

import flagquantum as fq

pytestmark = pytest.mark.braket


class FakeBraketTask:
    id = "braket-task-1"

    def state(self):
        return "COMPLETED"

    def result(self):
        return SimpleNamespace(measurement_counts={"00": 3, "01": 2})


class FakeAwsDevice:
    arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Fake"
    name = "IQM Fake"

    def __init__(self):
        self.properties = SimpleNamespace(
            paradigm=SimpleNamespace(
                qubitCount=4,
                nativeGateSet=("prx", "cz"),
                connectivity=SimpleNamespace(
                    connectivityGraph={
                        "0": ("1",),
                        "1": ("0", "2"),
                        "2": ("1", "3"),
                        "3": ("2",),
                    }
                ),
            ),
            provider=SimpleNamespace(providerName="IQM"),
        )
        self.runs = []

    def run(self, program, *, shots):
        self.runs.append((program, shots))
        return FakeBraketTask()


def _dynamic_package(provider):
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    return fq.experimental.create_dynamic_deployment_package(
        circuit,
        backend=provider.backend,
        shots=5,
    )


def test_braket_device_properties_build_dynamic_backend_profile() -> None:
    device = FakeAwsDevice()
    profile = fq.braket_backend_profile(
        device,
        dynamic_qubit_groups=((0, 1), (2, 3)),
    )

    assert profile.provider == "amazon-braket"
    assert profile.name == "IQM Fake"
    assert profile.n_wires == 4
    assert profile.basis_gates == ("prx", "cz")
    assert profile.coupling_map.edges == ((0, 1), (1, 2), (2, 3))
    assert profile.supports_dynamic_circuits is True
    assert profile.dynamic_dialect == "braket_iqm"
    assert profile.metadata["device_arn"] == device.arn
    assert profile.metadata["dynamic_qubit_groups"] == ((0, 1), (2, 3))


def test_braket_dry_run_does_not_submit_and_exposes_exact_program() -> None:
    device = FakeAwsDevice()
    provider = fq.AmazonBraketProvider(
        device,
        dynamic_qubit_groups=((0, 1), (2, 3)),
        program_factory=lambda *, source: {"source": source},
    )
    package = _dynamic_package(provider)

    preview = provider.dry_run(package)

    assert preview.compatible
    assert preview.program == package.qasm
    assert preview.shots == 5
    assert preview.summary()["dynamic_dialect"] == "braket_iqm"
    assert device.runs == []


def test_braket_provider_submits_program_and_normalizes_result() -> None:
    device = FakeAwsDevice()
    provider = fq.AmazonBraketProvider(
        device,
        dynamic_qubit_groups=((0, 1), (2, 3)),
        program_factory=lambda *, source: {"source": source},
    )
    package = _dynamic_package(provider)

    result = provider.run(package)

    assert device.runs == [({"source": package.qasm}, 5)]
    assert result.handle.task_id == "braket-task-1"
    assert result.counts == {"00": 3, "01": 2}
    assert result.shots == 5
    assert result.metadata["mid_circuit_measurements_returned"] is False
    assert (
        result.metadata["deployment_artifact_sha256"]
        == package.metadata["deployment_artifact_sha256"]
    )


def test_braket_iqm_missing_groups_fails_before_hardware_submission() -> None:
    device = FakeAwsDevice()
    provider = fq.AmazonBraketProvider(
        device,
        program_factory=lambda *, source: {"source": source},
    )
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    report = fq.experimental.assess_dynamic_backend(circuit, provider.backend)

    assert not report.compatible
    assert "braket_iqm_dynamic_qubit_groups_are_required" in report.blockers
    with pytest.raises(RuntimeError, match="dynamic_qubit_groups"):
        fq.experimental.create_dynamic_deployment_package(
            circuit,
            backend=provider.backend,
            shots=5,
        )
    assert device.runs == []
