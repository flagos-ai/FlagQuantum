"""Static validation for kernel semantic and implementation records."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .evidence import EVIDENCE
from .implementations import IMPLEMENTATIONS
from .schema import (
    KERNEL_DEVICES,
    KernelEvidence,
    KernelImplementation,
    KernelSemantic,
)
from .semantics import SEMANTICS

_CATALOG_ID = re.compile(r"^FQK-(SV|GR|MPS|NUM)-[0-9]{3}$")
_IMPLEMENTATION_ID = re.compile(
    r"^FQKI-(PYTORCH|TRITON|FLAGTREE)-(SV|GR|MPS|NUM)-[0-9]{3}-[A-Z]$"
)
_EVIDENCE_ID = re.compile(
    r"^FQKE-(PYTORCH|TRITON|FLAGTREE)-(SV|GR|MPS|NUM)-[0-9]{3}-[A-Z]$"
)
_TEST_REFERENCE = re.compile(
    r"^tests/(?:[a-z][a-z0-9_]*/)*test_[a-z0-9_]+\.py::test_[a-z0-9_]+$"
)
_ARTIFACT_REFERENCE = re.compile(r"^(?:artifacts|benchmarks/results)/[A-Za-z0-9_./-]+$")
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


def _require_devices(values: Sequence[str], label: str) -> None:
    _require_components(values, label)
    undeclared = sorted(set(values).difference(KERNEL_DEVICES))
    if undeclared:
        raise ValueError(
            f"undeclared {label}: {', '.join(undeclared)}; "
            f"declared devices are {', '.join(sorted(KERNEL_DEVICES))}"
        )


def validate_catalog(
    semantics: Sequence[KernelSemantic] = SEMANTICS,
    implementations: Sequence[KernelImplementation] = IMPLEMENTATIONS,
    evidence: Sequence[KernelEvidence] = EVIDENCE,
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
    _require_unique([item.evidence_id for item in evidence], "evidence ID")
    _require_unique(
        [item.implementation_id for item in evidence],
        "evidence implementation ID",
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
        _require_devices(
            implementation.devices,
            f"devices for {implementation.implementation_id}",
        )
        dimensions = (
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

    implementations_by_id = {item.implementation_id: item for item in implementations}
    evidenced_implementations: set[str] = set()
    for record in evidence:
        if not _EVIDENCE_ID.fullmatch(record.evidence_id):
            raise ValueError(f"invalid evidence ID: {record.evidence_id}")
        if record.implementation_id not in implementations_by_id:
            raise ValueError(
                f"unknown implementation ID for {record.evidence_id}: "
                f"{record.implementation_id}"
            )
        expected_evidence_id = record.implementation_id.replace("FQKI-", "FQKE-", 1)
        if record.evidence_id != expected_evidence_id:
            raise ValueError(
                f"evidence ID {record.evidence_id} does not match "
                f"implementation {record.implementation_id}"
            )
        if not record.correctness_tests:
            raise ValueError(
                f"missing correctness evidence for {record.implementation_id}"
            )
        implementation = implementations_by_id[record.implementation_id]
        if any(direction != "forward" for direction in implementation.directions):
            if not record.gradient_tests:
                raise ValueError(
                    f"missing gradient evidence for {record.implementation_id}"
                )
        if implementation.internal_fallback and not record.capability_tests:
            raise ValueError(
                f"missing fallback evidence for {record.implementation_id}"
            )
        test_references = (
            *record.correctness_tests,
            *record.gradient_tests,
            *record.capability_tests,
        )
        for references, label in (
            (record.correctness_tests, "correctness"),
            (record.gradient_tests, "gradient"),
            (record.capability_tests, "capability"),
        ):
            _require_unique(references, f"{label} evidence for {record.evidence_id}")
        invalid_tests = [
            reference
            for reference in test_references
            if _TEST_REFERENCE.fullmatch(reference) is None
        ]
        if invalid_tests:
            raise ValueError(
                f"invalid test evidence for {record.evidence_id}: "
                f"{', '.join(invalid_tests)}"
            )
        invalid_artifacts = [
            reference
            for reference in record.benchmark_artifacts
            if _ARTIFACT_REFERENCE.fullmatch(reference) is None
            or ".." in reference.split("/")
        ]
        if invalid_artifacts:
            raise ValueError(
                f"invalid benchmark evidence for {record.evidence_id}: "
                f"{', '.join(invalid_artifacts)}"
            )
        _require_components(
            record.required_lanes,
            f"required lanes for {record.evidence_id}",
        )
        evidenced_implementations.add(record.implementation_id)

    missing_evidence = sorted(
        set(implementations_by_id).difference(evidenced_implementations)
    )
    if missing_evidence:
        raise ValueError(
            f"implementations without evidence: {', '.join(missing_evidence)}"
        )


__all__ = ["validate_catalog"]
