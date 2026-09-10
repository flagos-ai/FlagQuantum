"""Auditable route and fallback enforcement for heterogeneous execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .fallback import FallbackPolicy


class RouteCategory(str, Enum):
    DEVICE_NATIVE = "device_native"
    DEVICE_PORTABLE = "device_portable"
    HOST = "host"


@dataclass(frozen=True)
class RouteExplanation:
    operator: str
    category: str
    provider: str
    device_type: str
    dtype: str
    route_manifest_hash: str
    details: str = ""

    def __post_init__(self) -> None:
        if self.category not in {item.value for item in RouteCategory}:
            raise ValueError(f"unsupported route category: {self.category!r}")
        if not all(
            (
                self.operator,
                self.provider,
                self.device_type,
                self.dtype,
                self.route_manifest_hash,
            )
        ):
            raise ValueError("route explanation identity fields must be non-empty")


@dataclass(frozen=True)
class FallbackEvent:
    operator: str
    source_route: str
    target_route: str
    reason: str
    host_transfer: bool = False
    dtype_demotion: bool = False

    def __post_init__(self) -> None:
        categories = {item.value for item in RouteCategory}
        if self.source_route not in categories or self.target_route not in categories:
            raise ValueError("fallback routes must use known route categories")
        if not self.operator or not self.reason:
            raise ValueError("fallback event must identify operator and reason")
        if self.host_transfer != (self.target_route == RouteCategory.HOST.value):
            raise ValueError(
                "host_transfer must be true exactly when target_route is host"
            )


@dataclass
class StrictExecutionScope:
    """Enforce one explicit fallback policy and retain an audit trail."""

    policy: FallbackPolicy | str = FallbackPolicy.FORBID
    allow_dtype_demotion: bool = False
    _routes: list[RouteExplanation] = field(default_factory=list, init=False)
    _fallbacks: list[FallbackEvent] = field(default_factory=list, init=False)
    _production_eligible: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.policy = FallbackPolicy.normalize(self.policy)

    @property
    def routes(self) -> tuple[RouteExplanation, ...]:
        return tuple(self._routes)

    @property
    def fallback_events(self) -> tuple[FallbackEvent, ...]:
        return tuple(self._fallbacks)

    @property
    def production_eligible(self) -> bool:
        return self._production_eligible

    def __enter__(self) -> "StrictExecutionScope":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def accept_route(self, route: RouteExplanation) -> None:
        policy = FallbackPolicy.normalize(self.policy)
        category = RouteCategory(route.category)
        if category is RouteCategory.HOST:
            if policy is not FallbackPolicy.HOST_DEBUG_ONLY:
                raise RuntimeError(
                    f"host route for {route.operator} is forbidden by "
                    f"{policy.value}"
                )
            self._production_eligible = False
        elif (
            category is RouteCategory.DEVICE_PORTABLE
            and policy is FallbackPolicy.FORBID
        ):
            raise RuntimeError(
                f"portable route for {route.operator} is forbidden by forbid policy"
            )
        self._routes.append(route)

    def record_fallback(self, event: FallbackEvent) -> None:
        policy = FallbackPolicy.normalize(self.policy)
        if policy is FallbackPolicy.FORBID:
            raise RuntimeError(
                f"fallback for {event.operator} is forbidden: {event.reason}"
            )
        if event.host_transfer and policy is not FallbackPolicy.HOST_DEBUG_ONLY:
            raise RuntimeError(
                f"host fallback for {event.operator} requires host_debug_only policy"
            )
        if event.dtype_demotion and not self.allow_dtype_demotion:
            raise RuntimeError(
                f"dtype demotion for {event.operator} is not in the precision plan"
            )
        if event.host_transfer:
            self._production_eligible = False
        self._fallbacks.append(event)


__all__ = (
    "FallbackEvent",
    "RouteCategory",
    "RouteExplanation",
    "StrictExecutionScope",
)
