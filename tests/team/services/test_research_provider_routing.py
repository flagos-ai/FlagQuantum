from __future__ import annotations

from dataclasses import dataclass

import pytest

from flagquantum.services import (
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    execute_research_capability,
    research_execution,
    validate_research_evidence,
)
from flagquantum.services.ground_state import GroundStateVQEResult, run_ground_state_vqe
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
class _FakeProvider:
    provider_id: str
    result: GroundStateVQEResult
    available: bool = True
    fail: bool = False
    calls: int = 0
    adapter_version: str = "test-only"

    def availability(self) -> ResearchProviderAvailability:
        return ResearchProviderAvailability(
            provider=self.provider_id,
            available=self.available,
            reason=None if self.available else "fixture dependency unavailable",
        )

    def estimate_backend_executions(self, request: GroundStateVQERequest) -> int:
        return request.optimizer.max_evaluations

    def run_vqe(self, request: GroundStateVQERequest) -> GroundStateVQEResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture provider execution failed")
        return self.result


def _install(monkeypatch: pytest.MonkeyPatch, *providers: _FakeProvider) -> None:
    monkeypatch.setattr(
        research_execution,
        "_installed_vqe_providers",
        lambda: tuple(providers),
    )


def test_explicit_provider_is_used_without_implicit_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    provider = _FakeProvider("fixture-primary", run_ground_state_vqe(request))
    _install(monkeypatch, provider)

    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {"provider_order": ["fixture-primary"]},
    )

    assert evidence.status == "succeeded"
    assert evidence.capability.provider == "fixture-primary"
    assert tuple(item.provider for item in evidence.attempts) == ("fixture-primary",)
    assert evidence.execution["provider_order"] == ["fixture-primary"]
    assert evidence.execution["fallback_used"] is False
    assert evidence.provenance["fallback_used"] is False
    route = evidence.metrics["provider_route"]
    assert route["selection_policy"] == "first_available_in_explicit_order"
    assert route["selected_provider"] == "fixture-primary"
    assert route["candidates"] == [
        {
            "provider": "fixture-primary",
            "rank": 1,
            "installed": True,
            "available": True,
            "score": 1,
            "selected": True,
            "reason": None,
            "preflight_error_code": None,
            "adapter_version": "test-only",
            "estimated_backend_executions": request.optimizer.max_evaluations,
        }
    ]
    assert provider.calls == 1
    assert validate_research_evidence("validate.energy_variance", evidence).passed


def test_multiple_providers_fail_closed_without_fallback_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    first = _FakeProvider("fixture-first", run_ground_state_vqe(request))
    second = _FakeProvider("fixture-second", first.result)
    _install(monkeypatch, first, second)

    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {"provider_order": ["fixture-first", "fixture-second"]},
    )

    assert evidence.status == "unsupported"
    assert evidence.error is not None
    assert evidence.error.code == "PROVIDER_FALLBACK_NOT_AUTHORIZED"
    assert evidence.attempts == ()
    assert (first.calls, second.calls) == (0, 0)


def test_authorized_unavailable_provider_falls_back_with_complete_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    unavailable = _FakeProvider(
        "fixture-unavailable", run_ground_state_vqe(request), available=False
    )
    fallback = _FakeProvider("fixture-fallback", unavailable.result)
    _install(monkeypatch, unavailable, fallback)

    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {
            "provider_order": ["fixture-unavailable", "fixture-fallback"],
            "allow_provider_fallback": True,
        },
    )

    assert evidence.status == "succeeded"
    assert evidence.capability.provider == "fixture-fallback"
    assert tuple((item.provider, item.status) for item in evidence.attempts) == (
        ("fixture-unavailable", "unsupported"),
        ("fixture-fallback", "succeeded"),
    )
    assert evidence.attempts[0].error is not None
    assert evidence.attempts[0].error.code == "PROVIDER_UNAVAILABLE"
    assert evidence.execution["fallback_used"] is True
    assert evidence.execution["call_counts"]["provider"] == 2
    assert evidence.provenance["fallback_used"] is True
    route = evidence.metrics["provider_route"]
    assert route["selected_provider"] == "fixture-fallback"
    assert [item["selected"] for item in route["candidates"]] == [False, True]
    assert [item["score"] for item in route["candidates"]] == [0, 1]
    assert route["candidates"][0]["preflight_error_code"] == "PROVIDER_UNAVAILABLE"
    assert (unavailable.calls, fallback.calls) == (0, 1)
    tampered = evidence.to_dict()
    tampered["attempts"][0]["provider"] = "hidden-provider"
    integrity = validate_research_evidence("validate.energy_variance", tampered)
    assert integrity.passed is False
    assert integrity.blockers[0].code == "EVIDENCE_INTEGRITY_MISMATCH"
    route_tampered = evidence.to_dict()
    route_tampered["metrics"]["provider_route"]["candidates"][0]["score"] = 99
    route_integrity = validate_research_evidence(
        "validate.energy_variance", route_tampered
    )
    assert route_integrity.passed is False
    assert route_integrity.blockers[0].code == "EVIDENCE_INTEGRITY_MISMATCH"


def test_unknown_provider_is_audited_and_can_only_fall_back_when_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    fallback = _FakeProvider("fixture-fallback", run_ground_state_vqe(request))
    _install(monkeypatch, fallback)

    denied = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {"provider_order": ["missing-provider"]},
    )
    allowed = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {
            "provider_order": ["missing-provider", "fixture-fallback"],
            "allow_provider_fallback": True,
        },
    )

    assert denied.status == "unsupported"
    assert denied.attempts[0].provider == "missing-provider"
    assert denied.attempts[0].error is not None
    assert denied.attempts[0].error.code == "PROVIDER_NOT_INSTALLED"
    assert allowed.status == "succeeded"
    assert tuple(item.provider for item in allowed.attempts) == (
        "missing-provider",
        "fixture-fallback",
    )
    assert allowed.execution["fallback_used"] is True


def test_execution_failure_does_not_trigger_unavailable_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    failing = _FakeProvider("fixture-failing", run_ground_state_vqe(request), fail=True)
    fallback = _FakeProvider("fixture-fallback", failing.result)
    _install(monkeypatch, failing, fallback)

    evidence = execute_research_capability(
        "ground_state.solve.vqe",
        request,
        {
            "provider_order": ["fixture-failing", "fixture-fallback"],
            "allow_provider_fallback": True,
        },
    )

    assert evidence.status == "failed"
    assert tuple(item.provider for item in evidence.attempts) == ("fixture-failing",)
    assert evidence.execution["fallback_used"] is False
    assert evidence.error is not None
    assert evidence.error.code == "CAPABILITY_EXECUTION_FAILED"
    assert (failing.calls, fallback.calls) == (1, 0)
