"""Closed execution policy for bounded research capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchExecutionOptions:
    """Validation thresholds, target, provider order, and fallback authority."""

    target: str = "local-cpu"
    exact_energy_tolerance: float = 2e-5
    energy_variance_tolerance: float = 1e-5
    provider_order: tuple[str, ...] = ("flagquantum-native",)
    allow_provider_fallback: bool = False


def validate_research_execution_options(
    options: ResearchExecutionOptions | Mapping[str, Any] | None,
) -> ResearchExecutionOptions:
    """Normalize public policy without silently widening any authority."""

    if options is None:
        value = ResearchExecutionOptions()
    elif isinstance(options, ResearchExecutionOptions):
        value = options
    elif isinstance(options, Mapping):
        allowed = {
            "target",
            "exact_energy_tolerance",
            "energy_variance_tolerance",
            "provider_order",
            "allow_provider_fallback",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown research execution option fields: {unknown}")
        values = dict(options)
        if "provider_order" in values and isinstance(
            values["provider_order"], (list, tuple)
        ):
            values["provider_order"] = tuple(values["provider_order"])
        value = ResearchExecutionOptions(**values)
    else:
        raise TypeError("options must be ResearchExecutionOptions, mapping, or None")
    for name, tolerance in (
        ("exact_energy_tolerance", value.exact_energy_tolerance),
        ("energy_variance_tolerance", value.energy_variance_tolerance),
    ):
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not 0 <= float(tolerance) < float("inf")
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if not isinstance(value.target, str) or not value.target:
        raise TypeError("target must be a non-empty string")
    if (
        not isinstance(value.provider_order, tuple)
        or not value.provider_order
        or any(
            not isinstance(provider, str) or not provider
            for provider in value.provider_order
        )
    ):
        raise TypeError("provider_order must be a non-empty sequence of provider IDs")
    if len(set(value.provider_order)) != len(value.provider_order):
        raise ValueError("provider_order must not contain duplicate provider IDs")
    if not isinstance(value.allow_provider_fallback, bool):
        raise TypeError("allow_provider_fallback must be a boolean")
    return ResearchExecutionOptions(
        target=value.target,
        exact_energy_tolerance=value.exact_energy_tolerance,
        energy_variance_tolerance=value.energy_variance_tolerance,
        provider_order=tuple(value.provider_order),
        allow_provider_fallback=value.allow_provider_fallback,
    )


__all__ = ("ResearchExecutionOptions", "validate_research_execution_options")
