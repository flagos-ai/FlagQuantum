"""Contract tests for the PennyLane Lightning research provider."""

from __future__ import annotations

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.ecosystem.pennylane.research_provider import (
    PennyLaneLightningResearchVQEProvider,
)
from flagquantum.services import (
    GroundStateOptimizer,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
)

pytestmark = pytest.mark.unit


def _request() -> GroundStateVQERequest:
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
        optimizer=GroundStateOptimizer(max_evaluations=81),
    )


def _native_statevector(circuit: Circuit) -> torch.Tensor:
    return circuit.state(refresh=True)


def test_provider_preflight_is_lazy_and_deterministic() -> None:
    provider = PennyLaneLightningResearchVQEProvider(
        _runner=lambda _circuit: (_ for _ in ()).throw(
            AssertionError("preflight must not execute a circuit")
        ),
        _probe=lambda: "fixture dependency unavailable",
    )

    availability = provider.availability()

    assert availability.provider == "pennylane-lightning"
    assert availability.available is False
    assert availability.reason == "fixture dependency unavailable"


def test_provider_uses_shared_parameter_shift_contract() -> None:
    request = _request()
    provider = PennyLaneLightningResearchVQEProvider(
        _runner=_native_statevector,
        _probe=lambda: None,
    )

    result = provider.run_vqe(request)

    assert result.status == "completed"
    assert result.energy == pytest.approx(-1.85727, abs=2e-3)
    assert result.resources is not None
    assert result.resources.objective_evaluations == 81
    assert result.resources.backend_executions == 241
    assert result.provenance.provider == "PennyLane Lightning"
    assert result.provenance.package == "pennylane-lightning"
