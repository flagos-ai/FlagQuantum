from unittest.mock import Mock

import pytest

import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.testing import InMemoryRemoteTarget

pytestmark = pytest.mark.unit


class _AzureTestTarget(InMemoryRemoteTarget):
    provider = "azure-quantum"

    def __init__(self, resource_id: str, target_id: str, *, n_wires: int) -> None:
        super().__init__()
        self.resource_id = resource_id
        self.backend = CloudBackendProfile(
            provider=self.provider,
            name=target_id,
            n_qubits=n_wires,
            metadata={
                "target_id": target_id,
                "device_provider": "test",
                "submission_format": "qir",
            },
        )


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AZURE_QUANTUM_RESOURCE_ID",
        "/subscriptions/test/resourceGroups/rg/providers/Microsoft.Quantum/"
        "Workspaces/workspace",
    )
    monkeypatch.setenv("AZURE_QUANTUM_TARGET_QUBITS", "20")
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", _AzureTestTarget
    )


def test_fq_run_azure_counts_preserves_stable_result_contract(monkeypatch) -> None:
    _configure(monkeypatch)

    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        outputs=fq.counts(name="samples"),
        target="azure:quantinuum.qpu.h2-1",
        shots=64,
        name="azure bell",
    )

    assert len(result.counts) == 1
    assert sum(result.counts[0].values()) == 64
    assert set(result.counts[0]).issubset({"00", "11"})
    assert result.measurement("samples").shots == 64
    assert result.provenance["provider"] == "azure-quantum"
    assert result.provenance["backend"] == "quantinuum.qpu.h2-1"
    assert result.provenance["compiler"] == "qdk"
    assert result.provenance["target"] == "azure:quantinuum.qpu.h2-1"
    assert result.provenance["target_qubits"] == ()
    assert result.provenance["name"] == "azure bell"
    assert result.runtime == {"mode": "remote_qpu", "shots": 64}
    native = result.native()
    assert native.handle.task_id == "local-1"


@pytest.mark.parametrize("compiler", ["qdk", "qsteed", "flagquantum", "QIR"])
def test_fq_run_azure_rejects_explicit_compiler_before_provider(
    monkeypatch: pytest.MonkeyPatch, compiler: str
) -> None:
    provider = Mock(side_effect=AssertionError("unexpected provider creation"))
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", provider
    )

    with pytest.raises(TypeError, match="does not accept compiler"):
        fq.run(
            fq.Circuit(1),
            compiler=compiler,
            target="azure:quantinuum.qpu.h2-1",
            shots=8,
        )
    provider.assert_not_called()


def test_fq_run_azure_rejects_expectation_before_provider(monkeypatch) -> None:
    provider = Mock(side_effect=AssertionError("unexpected provider creation"))
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", provider
    )

    with pytest.raises(ValueError, match="full-register fq.counts"):
        fq.run(
            fq.Circuit(1),
            outputs=fq.expectation(fq.Z(0)),
            target="azure:quantinuum.qpu.h2-1",
            shots=8,
        )
    provider.assert_not_called()


def test_fq_run_azure_rejects_physical_mapping_before_provider(monkeypatch) -> None:
    provider = Mock(side_effect=AssertionError("unexpected provider creation"))
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", provider
    )

    with pytest.raises(TypeError, match="does not accept target_qubits"):
        fq.run(
            fq.Circuit(1),
            target="azure:quantinuum.qpu.h2-1",
            target_qubits=(0,),
            shots=8,
        )
    provider.assert_not_called()


def test_fq_run_azure_requires_workspace_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_QUANTUM_RESOURCE_ID", raising=False)
    monkeypatch.setenv("AZURE_QUANTUM_TARGET_QUBITS", "20")
    provider = Mock(side_effect=AssertionError("unexpected provider creation"))
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", provider
    )

    with pytest.raises(RuntimeError, match="AZURE_QUANTUM_RESOURCE_ID"):
        fq.run(
            fq.Circuit(1),
            target="azure:quantinuum.qpu.h2-1",
            shots=8,
        )
    provider.assert_not_called()


def test_fq_run_azure_requires_authoritative_target_width_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_QUANTUM_RESOURCE_ID", "resource-id")
    monkeypatch.delenv("AZURE_QUANTUM_TARGET_QUBITS", raising=False)
    provider = Mock(side_effect=AssertionError("unexpected provider creation"))
    monkeypatch.setattr(
        "flagquantum.remote.qpu.execution.AzureQuantumProvider", provider
    )

    with pytest.raises(RuntimeError, match="AZURE_QUANTUM_TARGET_QUBITS"):
        fq.run(
            fq.Circuit(1),
            target="azure:quantinuum.qpu.h2-1",
            shots=8,
        )
    provider.assert_not_called()
