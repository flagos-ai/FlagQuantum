"""Static validation for kernel semantic and implementation records."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .implementations import IMPLEMENTATIONS
from .schema import KernelImplementation, KernelSemantic
from .semantics import SEMANTICS

_CATALOG_ID = re.compile(r"^FQK-(SV|GR|MPS|NUM)-[0-9]{3}$")
_IMPLEMENTATION_ID = re.compile(
    r"^FQKI-(PYTORCH|TRITON|FLAGTREE)-(SV|GR|MPS|NUM)-[0-9]{3}-[A-Z]$"
)
_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*$")
_DOMAIN_CODES = {
    "statevector": "SV",
    "gradient": "GR",
    "mps": "MPS",
    "numerics": "NUM",
}
_BANNED_SEMANTIC_COMPONENTS = {
    "complex64",
    "cpu",
    "cuda",
    "fast",
    "flagtree",
    "fused",
    "optimized",
    "qml",
    "triton",
    "v2",
}


def _require_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"duplicate {label}: {joined}")


def _require_components(values: Sequence[str], label: str) -> None:
    if not values or any(_COMPONENT.fullmatch(value) is None for value in values):
        raise ValueError(f"invalid or empty {label}")
    _require_unique(values, label)


def validate_catalog(
    semantics: Sequence[KernelSemantic] = SEMANTICS,
    implementations: Sequence[KernelImplementation] = IMPLEMENTATIONS,
) -> None:
    """Raise ``ValueError`` when catalog records violate catalog invariants."""

    _require_unique([item.catalog_id for item in semantics], "catalog ID")
    _require_unique([item.semantic_id for item in semantics], "semantic ID")
    _require_unique(
        [item.implementation_id for item in implementations],
        "implementation ID",
    )
    _require_unique(
        [f"{item.module}:{item.symbol}" for item in implementations],
        "implementation entry point",
    )

    semantics_by_id = {item.semantic_id: item for item in semantics}
    for semantic in semantics:
        if not _CATALOG_ID.fullmatch(semantic.catalog_id):
            raise ValueError(f"invalid catalog ID: {semantic.catalog_id}")
        expected_code = _DOMAIN_CODES[semantic.domain]
        if not semantic.catalog_id.startswith(f"FQK-{expected_code}-"):
            raise ValueError(
                f"catalog ID {semantic.catalog_id} does not match "
                f"domain {semantic.domain}"
            )
        components = semantic.semantic_id.split(".")
        if semantic.semantic_id.startswith(f"{semantic.domain}.") is False:
            raise ValueError(
                f"semantic ID {semantic.semantic_id} does not match "
                f"domain {semantic.domain}"
            )
        if any(_COMPONENT.fullmatch(component) is None for component in components):
            raise ValueError(f"invalid semantic ID: {semantic.semantic_id}")
        banned = _BANNED_SEMANTIC_COMPONENTS.intersection(components)
        if banned:
            joined = ", ".join(sorted(banned))
            raise ValueError(
                f"semantic ID {semantic.semantic_id} contains provider or "
                f"policy metadata: {joined}"
            )
        if not semantic.summary or not semantic.reference:
            raise ValueError(f"incomplete semantic record: {semantic.semantic_id}")
        _require_components(semantic.workloads, f"workload for {semantic.semantic_id}")

    implemented_semantics: set[str] = set()
    for implementation in implementations:
        if not _IMPLEMENTATION_ID.fullmatch(implementation.implementation_id):
            raise ValueError(
                f"invalid implementation ID: {implementation.implementation_id}"
            )
        if implementation.semantic_id not in semantics_by_id:
            raise ValueError(
                f"unknown semantic ID for {implementation.implementation_id}: "
                f"{implementation.semantic_id}"
            )
        semantic = semantics_by_id[implementation.semantic_id]
        catalog_fragment = semantic.catalog_id.removeprefix("FQK-")
        if f"-{catalog_fragment}-" not in implementation.implementation_id:
            raise ValueError(
                f"implementation ID {implementation.implementation_id} does not "
                f"match semantic {semantic.catalog_id}"
            )
        provider_token = implementation.provider.upper()
        if f"-{provider_token}-" not in implementation.implementation_id:
            raise ValueError(
                f"implementation ID {implementation.implementation_id} does not "
                f"match provider {implementation.provider}"
            )
        if not implementation.module.startswith("flagquantum.kernels."):
            raise ValueError(
                f"implementation module is outside kernel ownership: "
                f"{implementation.module}"
            )
        if _COMPONENT.fullmatch(implementation.symbol) is None:
            raise ValueError(f"invalid implementation symbol: {implementation.symbol}")
        dimensions = (
            (implementation.devices, "devices"),
            (implementation.dtypes, "dtypes"),
            (implementation.layouts, "layouts"),
            (implementation.directions, "directions"),
            (implementation.addressing, "addressing"),
        )
        for values, label in dimensions:
            _require_components(
                values,
                f"{label} for {implementation.implementation_id}",
            )
        implemented_semantics.add(implementation.semantic_id)

    missing = sorted(set(semantics_by_id).difference(implemented_semantics))
    if missing:
        raise ValueError(f"semantics without implementations: {', '.join(missing)}")


__all__ = ["validate_catalog"]
