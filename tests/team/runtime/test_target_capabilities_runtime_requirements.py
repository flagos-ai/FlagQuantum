"""Characterize Runtime matching needs without defining a public contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

import flagquantum as fq

SupportStatus = Literal["unknown", "unmeasured", "unsupported", "verified"]
ExposureStatus = Literal[
    "observed", "declared", "not_exposed", "unknown", "not_applicable"
]
RequirementSource = Literal["user", "compiler", "runtime_protocol"]
Comparison = Literal["equal", "at_least", "contains"]


@dataclass(frozen=True)
class _Requirement:
    name: str
    value: object
    source: RequirementSource = "compiler"
    comparison: Comparison = "equal"


@dataclass(frozen=True)
class _Fact:
    value: object
    support: SupportStatus
    exposure: ExposureStatus = "observed"
    mechanism: str | None = None


@dataclass(frozen=True)
class _Match:
    accepted: bool
    selected_device: str | None
    blockers: tuple[str, ...]
    fallback_events: tuple[str, ...] = ()


def _requirements_from_existing_plan(plan: object) -> tuple[_Requirement, ...]:
    """Test-only Compiler projection over the current stable plan payload."""

    payload = plan.to_dict()
    decision = payload["decision"]
    environment = payload["environment_requirements"]
    return (
        _Requirement("execution.mode", decision["mode"]),
        _Requirement("device.kind", environment["device_kind"]),
        _Requirement("precision.effective", decision["precision"]),
        _Requirement("device.count", decision["world_size"]),
        _Requirement("memory.limit_bytes", decision["memory_limit_bytes"]),
        _Requirement("gradient.required", decision["require_gradients"]),
        _Requirement("fallback.backend_allowed", decision["allow_backend_fallback"]),
    )


def _fact_satisfies(requirement: _Requirement, fact: _Fact) -> bool:
    if requirement.comparison == "equal":
        return fact.value == requirement.value
    if requirement.comparison == "at_least":
        return int(fact.value) >= int(requirement.value)
    if requirement.comparison == "contains":
        return set(requirement.value).issubset(set(fact.value))
    raise AssertionError(f"unhandled comparison: {requirement.comparison}")


def _match(requirements: tuple[_Requirement, ...], facts: dict[str, _Fact]) -> _Match:
    """Test fake for the proposed fail-closed Runtime policy seam."""

    blockers: list[str] = []
    requested_device = next(
        (item.value for item in requirements if item.name == "device.kind"), None
    )
    for requirement in requirements:
        if requirement.value is None:
            continue
        fact = facts.get(requirement.name)
        if fact is None:
            blockers.append(f"missing_fact:{requirement.name}")
            continue
        if fact.support == "unsupported":
            blockers.append(f"unsupported:{requirement.name}")
            continue
        if fact.support != "verified":
            blockers.append(f"not_verified:{requirement.name}:{fact.support}")
            continue
        if fact.exposure in {"not_exposed", "unknown"}:
            blockers.append(f"fact_not_exposed:{requirement.name}")
            continue
        if not _fact_satisfies(requirement, fact):
            blockers.append(f"mismatch:{requirement.name}")
    return _Match(
        accepted=not blockers,
        selected_device=str(requested_device) if not blockers else None,
        blockers=tuple(blockers),
    )


def _select_with_optional_cpu_fallback(
    requirements: tuple[_Requirement, ...],
    primary_facts: dict[str, _Fact],
    *,
    cpu_facts: dict[str, _Fact] | None = None,
    cpu_fallback_allowed: bool = False,
) -> _Match:
    primary = _match(requirements, primary_facts)
    if primary.accepted or not cpu_fallback_allowed or cpu_facts is None:
        return primary
    requested_device = next(
        (item.value for item in requirements if item.name == "device.kind"), None
    )
    cpu_requirements = tuple(
        _Requirement(
            item.name,
            "cpu" if item.name == "device.kind" else item.value,
            item.source,
            item.comparison,
        )
        for item in requirements
    )
    cpu = _match(cpu_requirements, cpu_facts)
    if not cpu.accepted:
        return _Match(
            accepted=False,
            selected_device=None,
            blockers=tuple(
                (*primary.blockers, *(f"cpu_candidate:{b}" for b in cpu.blockers))
            ),
        )
    return _Match(
        accepted=True,
        selected_device="cpu",
        blockers=(),
        fallback_events=(f"device:{requested_device}->cpu",),
    )


@pytest.mark.unit
def test_existing_plan_adapter_keeps_requirements_separate_from_facts() -> None:
    plan = fq.plan(
        fq.Circuit(2).h(0).cx(0, 1),
        options=fq.ExecutionOptions(
            mode="statevector",
            device="cpu",
            precision="complex128",
            memory_limit_bytes=4096,
            require_gradients=True,
            allow_backend_fallback=False,
        ),
    )

    requirements = dict(
        (item.name, item.value) for item in _requirements_from_existing_plan(plan)
    )

    assert requirements == {
        "execution.mode": "statevector",
        "device.kind": "cpu",
        "precision.effective": "complex128",
        "device.count": 1,
        "memory.limit_bytes": 4096,
        "gradient.required": True,
        "fallback.backend_allowed": False,
    }
    assert "device.available" not in requirements
    assert "memory.available_bytes" not in requirements
    assert {item.source for item in _requirements_from_existing_plan(plan)} == {
        "compiler"
    }


@pytest.mark.unit
@pytest.mark.parametrize("status", ["unknown", "unmeasured"])
def test_missing_or_unverified_platform_fact_fails_closed(
    status: SupportStatus,
) -> None:
    requirements = (_Requirement("topology.interconnect", "nvlink"),)
    facts = {
        "topology.interconnect": _Fact(
            "nvlink", status, "not_exposed" if status == "unknown" else "declared"
        )
    }

    result = _match(requirements, facts)

    assert result.accepted is False
    assert result.selected_device is None
    assert result.blockers


@pytest.mark.unit
def test_explicit_unsupported_is_not_weakened_by_runtime_policy() -> None:
    result = _match(
        (_Requirement("session.realtime", True),),
        {"session.realtime": _Fact(False, "unsupported", "declared")},
    )

    assert result.accepted is False
    assert result.blockers == ("unsupported:session.realtime",)


@pytest.mark.unit
def test_software_extended_precision_does_not_claim_native_fp64() -> None:
    facts = {
        "precision.effective": _Fact(
            "complex128", "verified", mechanism="double_single_software_extension"
        ),
        "precision.native": _Fact("complex128", "unsupported", "declared"),
    }

    effective = _match((_Requirement("precision.effective", "complex128"),), facts)
    native = _match((_Requirement("precision.native", "complex128"),), facts)

    assert effective.accepted is True
    assert native.accepted is False
    assert facts["precision.effective"].mechanism == "double_single_software_extension"


@pytest.mark.unit
def test_cpu_fallback_requires_authorization_and_emits_an_event() -> None:
    requirements = (
        _Requirement("device.kind", "cuda", source="user"),
        _Requirement("memory.available_bytes", 4096, comparison="at_least"),
    )
    cuda_facts = {
        "device.kind": _Fact("cuda", "unsupported", "declared"),
        "memory.available_bytes": _Fact(8192, "verified", "observed"),
    }
    cpu_facts = {
        "device.kind": _Fact("cpu", "verified", "observed"),
        "memory.available_bytes": _Fact(8192, "verified", "observed"),
    }

    forbidden = _select_with_optional_cpu_fallback(
        requirements,
        cuda_facts,
        cpu_facts=cpu_facts,
        cpu_fallback_allowed=False,
    )
    allowed = _select_with_optional_cpu_fallback(
        requirements,
        cuda_facts,
        cpu_facts=cpu_facts,
        cpu_fallback_allowed=True,
    )

    assert forbidden.accepted is False
    assert forbidden.fallback_events == ()
    assert allowed.accepted is True
    assert allowed.selected_device == "cpu"
    assert allowed.fallback_events == ("device:cuda->cpu",)


@pytest.mark.unit
def test_backend_fallback_permission_does_not_authorize_cpu_fallback() -> None:
    result = _select_with_optional_cpu_fallback(
        (
            _Requirement("device.kind", "cuda", source="user"),
            _Requirement("fallback.backend_allowed", True, source="user"),
        ),
        {
            "device.kind": _Fact("cuda", "unsupported", "declared"),
            "fallback.backend_allowed": _Fact(True, "verified", "declared"),
        },
        cpu_facts={
            "device.kind": _Fact("cpu", "verified", "observed"),
            "fallback.backend_allowed": _Fact(True, "verified", "declared"),
        },
        cpu_fallback_allowed=False,
    )

    assert result.accepted is False
    assert result.selected_device is None
    assert result.fallback_events == ()


@pytest.mark.unit
def test_cpu_fallback_rematches_every_requirement_against_cpu_snapshot() -> None:
    result = _select_with_optional_cpu_fallback(
        (
            _Requirement("device.kind", "cuda", source="user"),
            _Requirement("memory.available_bytes", 4096, comparison="at_least"),
        ),
        {"device.kind": _Fact("cuda", "unsupported", "declared")},
        cpu_facts={"device.kind": _Fact("cpu", "verified", "observed")},
        cpu_fallback_allowed=True,
    )

    assert result.accepted is False
    assert result.fallback_events == ()
    assert "cpu_candidate:missing_fact:memory.available_bytes" in result.blockers


@pytest.mark.unit
def test_not_exposed_route_cannot_prove_absence_of_cpu_fallback() -> None:
    result = _match(
        (_Requirement("route.cpu_fallback_observed", False),),
        {
            "route.cpu_fallback_observed": _Fact(
                False, "verified", exposure="not_exposed"
            )
        },
    )

    assert result.accepted is False
    assert result.blockers == ("fact_not_exposed:route.cpu_fallback_observed",)
