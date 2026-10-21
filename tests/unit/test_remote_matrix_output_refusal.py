"""Remote-specific bypasses refuse proposal 080 matrices before any submission."""

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.remote import jobs
from flagquantum.remote.qpu import execution

pytestmark = pytest.mark.unit


@pytest.fixture
def matrix_request() -> fq.OutputRequest:
    return fq.expectation(torch.eye(2), name="population")


@pytest.mark.parametrize(
    "validator", [execution.validate_quafu_output, execution.validate_azure_output]
)
@pytest.mark.parametrize("multiple", [False, True])
def test_remote_validators_refuse_every_matrix_before_kind_dispatch(
    matrix_request: fq.OutputRequest,
    validator: Callable[..., fq.OutputRequest],
    multiple: bool,
) -> None:
    outputs = [fq.counts(), matrix_request] if multiple else matrix_request
    with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
        validator(fq.Circuit(1), outputs)


def test_valid_pauli_remote_contracts_keep_their_existing_behavior() -> None:
    output = fq.expectation(fq.Z(0))
    assert execution.validate_quafu_output(fq.Circuit(1), output) is output
    with pytest.raises(ValueError, match="full-register"):
        execution.validate_azure_output(fq.Circuit(1), output)
    with pytest.raises(ValueError, match="Hamiltonian references qubits"):
        execution.validate_quafu_output(fq.Circuit(1), fq.expectation(fq.Z(1)))


@pytest.mark.parametrize("provider", ["quafu", "azure"])
def test_direct_remote_execution_never_builds_a_package_or_provider(
    matrix_request: fq.OutputRequest,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported matrix output reached remote preparation")

    for name in ("create_deployment_package", "QuafuProvider", "AzureQuantumProvider"):
        monkeypatch.setattr(execution, name, unexpected)
    function = (
        execution.execute_quafu if provider == "quafu" else execution.execute_azure
    )
    kwargs = {"compiler": None} if provider == "quafu" else {}
    with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
        function(
            fq.Circuit(1).to_ir(),
            output=matrix_request,
            target=f"{provider}:test",
            shots=100,
            name=None,
            **kwargs,
        )


def test_detached_quafu_job_never_submits_or_writes_receipt(
    matrix_request: fq.OutputRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported matrix output reached provider construction")

    monkeypatch.setattr(jobs, "QuafuProvider", unexpected)
    with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
        jobs.submit(
            fq.Circuit(1), target="quafu:test", outputs=matrix_request, shots=100
        )
