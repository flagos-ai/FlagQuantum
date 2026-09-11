from __future__ import annotations

import pytest

from flagquantum.runtime.routing import (
    FallbackEvent,
    RouteExplanation,
    StrictExecutionScope,
)


def _route(category: str) -> RouteExplanation:
    return RouteExplanation(
        operator="aten::bmm",
        category=category,
        provider="torch_fl",
        device_type="flagos",
        dtype="complex64",
        route_manifest_hash="sha256:test",
    )


def test_forbid_policy_accepts_only_device_native_route() -> None:
    scope = StrictExecutionScope("forbid")
    scope.accept_route(_route("device_native"))

    with pytest.raises(RuntimeError, match="portable route"):
        scope.accept_route(_route("device_portable"))


def test_same_device_portable_rejects_host_transfer() -> None:
    scope = StrictExecutionScope("same_device_portable")
    scope.accept_route(_route("device_portable"))

    with pytest.raises(RuntimeError, match="host fallback"):
        scope.record_fallback(
            FallbackEvent(
                operator="aten::bmm",
                source_route="device_portable",
                target_route="host",
                reason="operator unavailable",
                host_transfer=True,
            )
        )


def test_host_debug_route_disables_production_claim() -> None:
    scope = StrictExecutionScope("host_debug_only")
    scope.accept_route(_route("host"))

    assert not scope.production_eligible


def test_dtype_demotion_requires_precision_plan_authorization() -> None:
    event = FallbackEvent(
        operator="aten::bmm",
        source_route="device_native",
        target_route="device_portable",
        reason="complex128 kernel unavailable",
        dtype_demotion=True,
    )

    with pytest.raises(RuntimeError, match="not in the precision plan"):
        StrictExecutionScope("same_device_portable").record_fallback(event)

    allowed = StrictExecutionScope("same_device_portable", allow_dtype_demotion=True)
    allowed.record_fallback(event)
    assert allowed.fallback_events == (event,)


def test_reassigned_string_policy_still_forbids_portable_execution() -> None:
    scope = StrictExecutionScope("same_device_portable")
    scope.policy = "forbid"
    with pytest.raises(RuntimeError, match="portable route"):
        scope.accept_route(_route("device_portable"))
    with pytest.raises(RuntimeError, match="fallback.*forbidden"):
        scope.record_fallback(
            FallbackEvent(
                operator="aten::bmm",
                source_route="device_native",
                target_route="device_portable",
                reason="native kernel unavailable",
            )
        )
    assert scope.routes == ()
    assert scope.fallback_events == ()


def test_reassigned_string_policy_preserves_host_debug_audit() -> None:
    scope = StrictExecutionScope()
    scope.policy = "host_debug_only"
    scope.accept_route(_route("host"))
    event = FallbackEvent(
        operator="aten::bmm",
        source_route="device_native",
        target_route="host",
        reason="host diagnostic requested",
        host_transfer=True,
    )
    scope.record_fallback(event)
    assert scope.routes == (_route("host"),)
    assert scope.fallback_events == (event,)
    assert not scope.production_eligible
