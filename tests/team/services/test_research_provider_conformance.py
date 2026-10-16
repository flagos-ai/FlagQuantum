"""Native and replaceable provider-function conformance for the P0 service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from flagquantum.services import (
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    execute_research_capability,
    research_execution,
)
from flagquantum.services.ground_state import (
    EnergyVarianceReference,
    ExactSectorReference,
    GroundStateVQEResult,
    StatevectorEvidence,
    energy_variance_reference,
    exact_sector_reference,
    run_ground_state_vqe,
)
from flagquantum.services.research_providers import ResearchProviderAvailability

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
    )


@dataclass
class _FixtureProvider:
    """Function-level fake for the exact dependency seam used by the service."""

    request: GroundStateVQERequest
    vqe_result: GroundStateVQEResult
    exact_result: ExactSectorReference
    variance_result: EnergyVarianceReference
    vqe_calls: int = 0
    exact_calls: int = 0
    variance_calls: int = 0

    @property
    def provider_id(self) -> str:
        return "flagquantum-native"

    @property
    def adapter_version(self) -> str:
        return "test-only"

    def availability(self) -> ResearchProviderAvailability:
        return ResearchProviderAvailability(self.provider_id, True)

    def estimate_backend_executions(self, request: GroundStateVQERequest) -> int:
        return request.optimizer.max_evaluations

    def run_vqe(self, request: GroundStateVQERequest) -> GroundStateVQEResult:
        assert request == self.request
        self.vqe_calls += 1
        return self.vqe_result

    def exact(
        self, hamiltonian: PauliSum, *, fixed_hamming_weight: int
    ) -> ExactSectorReference:
        assert hamiltonian == self.request.hamiltonian
        assert fixed_hamming_weight == self.request.fixed_hamming_weight
        self.exact_calls += 1
        return self.exact_result

    def variance(
        self,
        hamiltonian: PauliSum,
        statevector: StatevectorEvidence,
        *,
        tolerance: float,
    ) -> EnergyVarianceReference:
        assert hamiltonian == self.request.hamiltonian
        assert statevector == self.vqe_result.statevector
        assert tolerance == pytest.approx(1e-5)
        self.variance_calls += 1
        return self.variance_result


def _fixture_provider(request: GroundStateVQERequest) -> _FixtureProvider:
    vqe_result = run_ground_state_vqe(request)
    assert vqe_result.statevector is not None
    exact_result = exact_sector_reference(
        request.hamiltonian,
        fixed_hamming_weight=request.fixed_hamming_weight or 0,
    )
    variance_result = energy_variance_reference(
        request.hamiltonian,
        vqe_result.statevector,
        tolerance=1e-5,
    )
    return _FixtureProvider(request, vqe_result, exact_result, variance_result)


def _semantic_projection(evidence: Any) -> dict[str, Any]:
    metrics = dict(evidence.metrics)
    metrics.pop("provider_route", None)
    return {
        "capability": evidence.capability,
        "status": evidence.status,
        "result": evidence.result,
        "metrics": metrics,
        "validation": evidence.validation,
        "warnings": evidence.warnings,
        "error": evidence.error,
        "call_counts": evidence.execution["call_counts"],
    }


def test_native_and_provider_fake_satisfy_the_same_service_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    native = execute_research_capability("ground_state.solve.vqe", request)
    provider = _fixture_provider(request)
    patches: tuple[tuple[str, Callable[..., object]], ...] = (
        ("exact_sector_reference", provider.exact),
        ("energy_variance_reference", provider.variance),
    )
    for name, replacement in patches:
        monkeypatch.setattr(research_execution, name, replacement)
    monkeypatch.setattr(
        research_execution, "_installed_vqe_providers", lambda: (provider,)
    )

    replaced = execute_research_capability("ground_state.solve.vqe", request)

    assert _semantic_projection(replaced) == _semantic_projection(native)
    assert (provider.vqe_calls, provider.exact_calls, provider.variance_calls) == (
        1,
        1,
        1,
    )
    assert replaced.execution["fallback_used"] is False
    assert replaced.provenance["fallback_used"] is False


def test_provider_failure_is_mapped_without_leaking_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingProvider:
        provider_id = "flagquantum-native"
        adapter_version = "test-only"

        def availability(self) -> ResearchProviderAvailability:
            return ResearchProviderAvailability(self.provider_id, True)

        def estimate_backend_executions(self, request: GroundStateVQERequest) -> int:
            return request.optimizer.max_evaluations

        def run_vqe(self, _request: GroundStateVQERequest) -> GroundStateVQEResult:
            raise RuntimeError("fixture provider unavailable")

    monkeypatch.setattr(
        research_execution,
        "_installed_vqe_providers",
        lambda: (_FailingProvider(),),
    )

    evidence = execute_research_capability("ground_state.solve.vqe", _request())

    assert evidence.status == "failed"
    assert evidence.error is not None
    assert evidence.error.code == "CAPABILITY_EXECUTION_FAILED"
    assert evidence.error.stage == "execution"
    assert evidence.error.context == {"exception_type": "RuntimeError"}
    assert evidence.attempts[0].status == "failed"
    assert evidence.execution["fallback_used"] is False
