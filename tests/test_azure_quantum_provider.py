from types import SimpleNamespace

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
from flagquantum.remote import AzureQuantumProvider, azure_backend_profile

pytestmark = pytest.mark.unit


class FakeAzureJob:
    id = "azure-job-1"

    def __init__(self) -> None:
        self.details = SimpleNamespace(status="Succeeded", shots=5)
        self.cancelled = False

    def refresh(self) -> None:
        return None

    def get_results(self):
        return {"probabilities": {"[0, 0]": 0.6, "[1, 1]": 0.4}}

    def cancel(self):
        self.cancelled = True
        return True


class FakeAzureTarget:
    name = "quantinuum.qpu.h2-1"
    provider_id = "quantinuum"

    def __init__(self) -> None:
        self.submissions = []
        self.job = FakeAzureJob()

    def submit(self, program, name, *, shots):
        self.submissions.append((program, name, shots))
        return self.job


class FakeAzureWorkspace:
    def __init__(self, target) -> None:
        self.target = target
        self.requested_targets = []

    def get_targets(self, name):
        self.requested_targets.append(name)
        return self.target

    def get_job(self, job_id):
        assert job_id == self.target.job.id
        return self.target.job


def _provider():
    target = FakeAzureTarget()
    workspace = FakeAzureWorkspace(target)
    compiled = []

    def compile_qasm(source):
        compiled.append(source)
        return {"qir": source}

    provider = AzureQuantumProvider(
        workspace,
        target.name,
        n_wires=32,
        basis_gates=("rz", "rzz"),
        program_factory=compile_qasm,
    )
    return provider, target, compiled


def _package(provider):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    return fqd.create_deployment_package(
        circuit,
        backend=provider.backend,
        shots=5,
        qasm_version=3.0,
    )


def test_azure_target_builds_explicit_fail_closed_profile() -> None:
    target = FakeAzureTarget()

    profile = azure_backend_profile(
        target,
        n_wires=32,
        basis_gates=("RZ", "RZZ"),
    )

    assert profile.provider == "azure-quantum"
    assert profile.name == target.name
    assert profile.n_wires == 32
    assert profile.basis_gates == ("rz", "rzz")
    assert profile.supports_dynamic_circuits is False
    assert profile.metadata["device_provider"] == "quantinuum"
    assert profile.metadata["submission_format"] == "qir"


def test_azure_dry_run_does_not_compile_or_submit() -> None:
    provider, target, compiled = _provider()
    package = _package(provider)

    preview = provider.dry_run(package)

    assert preview.compatible
    assert preview.program == package.qasm
    assert preview.summary()["submission_format"] == "qir"
    assert compiled == []
    assert target.submissions == []


def test_azure_provider_submits_qir_and_converts_probabilities_to_counts() -> None:
    provider, target, compiled = _provider()
    package = _package(provider)

    result = provider.run(package)

    assert compiled == [package.qasm]
    assert target.submissions == [({"qir": package.qasm}, package.name, 5)]
    assert result.handle.task_id == "azure-job-1"
    assert result.counts == {"00": 3, "11": 2}
    assert result.shots == 5
    assert result.metadata["submission_format"] == "qir"
    assert (
        result.metadata["deployment_artifact_sha256"]
        == package.metadata["deployment_artifact_sha256"]
    )


def test_azure_status_and_cancellation_use_job_lifecycle() -> None:
    provider, target, _compiled = _provider()
    handle = provider.submit(_package(provider))

    assert provider.query_status(handle) == "Finished"
    assert provider.cancel(handle) is True
    assert target.job.cancelled is True


def test_azure_preflight_rejects_openqasm_2_before_compilation() -> None:
    provider, target, compiled = _provider()
    package = fqd.create_deployment_package(
        fq.Circuit(2).h(0).cx(0, 1),
        backend=provider.backend,
        shots=5,
        qasm_version=2.0,
    )

    preview = provider.dry_run(package)

    assert not preview.compatible
    assert "azure_quantum_provider_requires_openqasm_3" in preview.blockers
    with pytest.raises(RuntimeError, match="requires_openqasm_3"):
        provider.submit(package)
    assert compiled == []
    assert target.submissions == []
