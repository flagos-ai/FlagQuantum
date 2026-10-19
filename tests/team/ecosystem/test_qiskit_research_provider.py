"""Contract tests for the Qiskit Aer research-provider adapter."""

from __future__ import annotations

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.ecosystem.qiskit.research_provider import (
    QiskitAerResearchVQEProvider,
)
from flagquantum.services import (
    GroundStateOptimizer,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
)
from flagquantum.services.research_providers import installed_vqe_providers

pytestmark = pytest.mark.unit


def _request(*, max_evaluations: int = 81) -> GroundStateVQERequest:
    return GroundStateVQERequest(
        hamiltonian=PauliSum(
            n_qubits=2,
            terms=(
                PauliSumTerm(-1.052373245772859, "II"),
                PauliSumTerm(0.39793742484318045, "ZI"),
                PauliSumTerm(-0.39793742484318045, "IZ"),
                PauliSumTerm(-0.01128010425623538, "ZZ"),
                PauliSumTerm(0.18093119978423156, "XX"),
            ),
        ),
        fixed_hamming_weight=1,
        optimizer=GroundStateOptimizer(max_evaluations=max_evaluations),
    )


def _native_statevector(circuit: Circuit) -> torch.Tensor:
    return circuit.state(refresh=True)


def test_provider_is_discoverable_without_importing_optional_dependencies() -> None:
    providers = {
        provider.provider_id: provider for provider in installed_vqe_providers()
    }

    assert set(providers) == {
        "flagquantum-native",
        "qiskit-aer",
        "pennylane-lightning",
    }
    availability = providers["qiskit-aer"].availability()
    assert availability.provider == "qiskit-aer"
    assert isinstance(availability.available, bool)
    assert availability.reason is None or availability.reason.startswith(
        "missing optional dependency:"
    )


def test_provider_reports_preflight_reason_without_calling_runner() -> None:
    calls = 0

    def runner(_circuit: Circuit) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise AssertionError("runner must not be called during preflight")

    provider = QiskitAerResearchVQEProvider(
        _runner=runner,
        _probe=lambda: "fixture dependency unavailable",
    )

    assert provider.availability().available is False
    assert provider.availability().reason == "fixture dependency unavailable"
    assert calls == 0


def test_parameter_shift_adam_returns_provider_owned_evidence() -> None:
    request = _request()
    provider = QiskitAerResearchVQEProvider(
        _runner=_native_statevector,
        _probe=lambda: None,
    )

    result = provider.run_vqe(request)

    assert result.status == "completed"
    assert result.energy == pytest.approx(-1.85727, abs=2e-3)
    assert len(result.convergence_trace) == request.optimizer.max_evaluations
    assert result.resources is not None
    assert result.resources.optimizer_steps == request.optimizer.max_evaluations - 1
    assert result.resources.objective_evaluations == request.optimizer.max_evaluations
    assert result.resources.backend_executions == (
        3 * (request.optimizer.max_evaluations - 1) + 1
    )
    assert result.provenance.provider == "Qiskit Aer"
    assert result.provenance.package == "qiskit-aer"
    assert result.provenance.execution_digest.startswith("sha256:")
    assert result.statevector is not None
    assert result.statevector.norm == pytest.approx(1.0, abs=1e-6)
