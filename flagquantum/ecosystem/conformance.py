"""Reusable, dependency-free conformance checks for interop adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from ..core.ir import CircuitIR, ensure_circuit_ir
from .contracts import (
    INTEROP_API_VERSION,
    InteropAdapter,
    InteropConversionError,
    InteropConversionReport,
    InteropExportResult,
    InteropImportResult,
)

InteropDirection = Literal["import", "export"]
SemanticFingerprint = Callable[[Any], str]


@dataclass(frozen=True)
class InteropConformanceViolation:
    """One stable, machine-readable adapter contract violation."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("conformance violation code and message must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class InteropRoundTripCase:
    """One FlagQuantum IR program that an adapter must round-trip losslessly."""

    name: str
    program: CircuitIR

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("interop conformance case name must be non-empty")
        object.__setattr__(self, "program", ensure_circuit_ir(self.program))


@dataclass(frozen=True)
class InteropRejectionCase:
    """One unsupported value that must fail closed before explicit lossy use."""

    name: str
    direction: InteropDirection
    value: Any
    expected_issue_code: str
    lossy_supported: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.expected_issue_code:
            raise ValueError("rejection case name and issue code must be non-empty")
        if self.direction not in {"import", "export"}:
            raise ValueError(f"unknown interop direction {self.direction!r}")


@dataclass(frozen=True)
class InteropConformanceCaseResult:
    """Outcome of one round-trip or rejection case."""

    name: str
    kind: Literal["round_trip", "rejection"]
    passed: bool
    violations: tuple[InteropConformanceViolation, ...] = ()
    source_fingerprint: str | None = None
    round_trip_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "source_fingerprint": self.source_fingerprint,
            "round_trip_fingerprint": self.round_trip_fingerprint,
            "violations": [item.to_dict() for item in self.violations],
        }


@dataclass(frozen=True)
class InteropConformanceResult:
    """Aggregate framework-neutral adapter certification result."""

    adapter: str
    api_version: str
    cases: tuple[InteropConformanceCaseResult, ...]
    violations: tuple[InteropConformanceViolation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations and all(case.passed for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_interop_conformance_v1",
            "adapter": self.adapter,
            "api_version": self.api_version,
            "passed": self.passed,
            "violations": [item.to_dict() for item in self.violations],
            "cases": [case.to_dict() for case in self.cases],
        }


def semantic_fingerprint(
    program: Any,
    *,
    semantic_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash executable IR semantics while excluding transport provenance."""

    ir = ensure_circuit_ir(program)
    encoded = ir.to_dict()
    payload = {
        "schema": "flagquantum_interop_semantics_v1",
        "ir_version": ir.version,
        "n_wires": ir.n_wires,
        "instructions": encoded["instructions"],
        "observables": encoded["observables"],
        "measurements": encoded["measurements"],
        "semantic_metadata": dict(semantic_metadata or {}),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _violation(code: str, message: str) -> InteropConformanceViolation:
    return InteropConformanceViolation(code, message)


def _report_violations(
    adapter: InteropAdapter,
    report: Any,
    *,
    require_lossless: bool,
) -> list[InteropConformanceViolation]:
    if not isinstance(report, InteropConversionReport):
        return [
            _violation("invalid_report_type", "conversion returned no interop report")
        ]
    violations: list[InteropConformanceViolation] = []
    if report.adapter != adapter.name:
        violations.append(
            _violation(
                "report_adapter_mismatch",
                f"report adapter {report.adapter!r} does not match {adapter.name!r}",
            )
        )
    if require_lossless and not report.lossless:
        violations.append(
            _violation(
                "unexpected_semantic_loss", "lossless case reported semantic loss"
            )
        )
    return violations


def _round_trip_case(
    adapter: InteropAdapter,
    case: InteropRoundTripCase,
    fingerprint: SemanticFingerprint,
) -> InteropConformanceCaseResult:
    violations: list[InteropConformanceViolation] = []
    source_fingerprint = fingerprint(case.program)
    round_trip_fingerprint: str | None = None
    try:
        exported = adapter.export_program(case.program)
        if not isinstance(exported, InteropExportResult):
            violations.append(
                _violation(
                    "invalid_export_result", "export did not return InteropExportResult"
                )
            )
        else:
            violations.extend(
                _report_violations(adapter, exported.report, require_lossless=True)
            )
            imported = adapter.import_program(exported.artifact)
            if not isinstance(imported, InteropImportResult):
                violations.append(
                    _violation(
                        "invalid_import_result",
                        "import did not return InteropImportResult",
                    )
                )
            else:
                violations.extend(
                    _report_violations(adapter, imported.report, require_lossless=True)
                )
                round_trip_fingerprint = fingerprint(imported.ir)
                if source_fingerprint != round_trip_fingerprint:
                    violations.append(
                        _violation(
                            "semantic_round_trip_mismatch",
                            "FlagQuantum IR semantics changed during export/import",
                        )
                    )
    except Exception as exc:  # adapter exceptions are conformance evidence
        violations.append(
            _violation(
                "round_trip_raised",
                f"{type(exc).__name__}: {exc}",
            )
        )
    return InteropConformanceCaseResult(
        name=case.name,
        kind="round_trip",
        passed=not violations,
        violations=tuple(violations),
        source_fingerprint=source_fingerprint,
        round_trip_fingerprint=round_trip_fingerprint,
    )


def _convert(
    adapter: InteropAdapter,
    direction: InteropDirection,
    value: Any,
    *,
    allow_lossy: bool,
) -> InteropImportResult | InteropExportResult:
    if direction == "import":
        return adapter.import_program(value, allow_lossy=allow_lossy)
    return adapter.export_program(value, allow_lossy=allow_lossy)


def _has_issue(report: InteropConversionReport, code: str) -> bool:
    return any(issue.code == code for issue in report.issues)


def _rejection_case(
    adapter: InteropAdapter,
    case: InteropRejectionCase,
) -> InteropConformanceCaseResult:
    violations: list[InteropConformanceViolation] = []
    try:
        _convert(adapter, case.direction, case.value, allow_lossy=False)
    except InteropConversionError as exc:
        violations.extend(
            _report_violations(adapter, exc.report, require_lossless=False)
        )
        if not _has_issue(exc.report, case.expected_issue_code):
            violations.append(
                _violation(
                    "missing_expected_issue",
                    f"strict rejection did not report {case.expected_issue_code!r}",
                )
            )
    except Exception as exc:
        violations.append(
            _violation(
                "wrong_rejection_error",
                f"strict rejection raised {type(exc).__name__}, not InteropConversionError",
            )
        )
    else:
        violations.append(
            _violation(
                "strict_rejection_accepted", "unsupported value did not fail closed"
            )
        )

    try:
        lossy_result = _convert(adapter, case.direction, case.value, allow_lossy=True)
    except InteropConversionError as exc:
        violations.extend(
            _report_violations(adapter, exc.report, require_lossless=False)
        )
        if case.lossy_supported:
            violations.append(
                _violation(
                    "lossy_conversion_rejected",
                    f"explicit lossy conversion raised {type(exc).__name__}: {exc}",
                )
            )
        elif not _has_issue(exc.report, case.expected_issue_code):
            violations.append(
                _violation(
                    "missing_expected_issue",
                    f"lossy rejection did not report {case.expected_issue_code!r}",
                )
            )
    except Exception as exc:
        violations.append(
            _violation("lossy_conversion_raised", f"{type(exc).__name__}: {exc}")
        )
    else:
        if not case.lossy_supported:
            violations.append(
                _violation(
                    "unexpected_lossy_acceptance",
                    "lossy conversion should remain blocked",
                )
            )
        report = lossy_result.report
        violations.extend(_report_violations(adapter, report, require_lossless=False))
        if report.lossless:
            violations.append(
                _violation(
                    "unreported_semantic_loss", "lossy conversion reported lossless"
                )
            )
        if not _has_issue(report, case.expected_issue_code):
            violations.append(
                _violation(
                    "missing_expected_issue",
                    f"lossy conversion did not report {case.expected_issue_code!r}",
                )
            )
    return InteropConformanceCaseResult(
        name=case.name,
        kind="rejection",
        passed=not violations,
        violations=tuple(violations),
    )


def run_adapter_conformance(
    adapter: Any,
    round_trip_cases: Sequence[InteropRoundTripCase],
    *,
    rejection_cases: Sequence[InteropRejectionCase] = (),
    fingerprint: SemanticFingerprint | None = None,
) -> InteropConformanceResult:
    """Run the common identity, round-trip, and fail-closed adapter contract."""

    fingerprint = semantic_fingerprint if fingerprint is None else fingerprint
    names = [case.name for case in round_trip_cases] + [
        case.name for case in rejection_cases
    ]
    if len(names) != len(set(names)):
        raise ValueError("interop conformance case names must be unique")
    violations: list[InteropConformanceViolation] = []
    if not round_trip_cases:
        violations.append(
            _violation(
                "missing_round_trip_cases",
                "adapter conformance requires at least one lossless round-trip case",
            )
        )
    adapter_name = str(getattr(adapter, "name", ""))
    api_version = str(getattr(adapter, "api_version", ""))
    dependency_extra = str(getattr(adapter, "dependency_extra", ""))
    conforms = isinstance(adapter, InteropAdapter)
    if not conforms:
        violations.append(
            _violation("invalid_adapter", "object does not implement InteropAdapter")
        )
    if api_version != INTEROP_API_VERSION:
        violations.append(
            _violation(
                "api_version_mismatch",
                f"adapter targets {api_version!r}, expected {INTEROP_API_VERSION!r}",
            )
        )
    if not adapter_name or not dependency_extra:
        violations.append(
            _violation(
                "invalid_adapter_identity", "adapter identity fields must be non-empty"
            )
        )
    results: tuple[InteropConformanceCaseResult, ...] = ()
    if conforms:
        results = tuple(
            [_round_trip_case(adapter, case, fingerprint) for case in round_trip_cases]
            + [_rejection_case(adapter, case) for case in rejection_cases]
        )
    return InteropConformanceResult(
        adapter=adapter_name,
        api_version=api_version,
        cases=results,
        violations=tuple(violations),
    )


__all__ = (
    "InteropConformanceCaseResult",
    "InteropConformanceResult",
    "InteropConformanceViolation",
    "InteropRejectionCase",
    "InteropRoundTripCase",
    "SemanticFingerprint",
    "run_adapter_conformance",
    "semantic_fingerprint",
)
