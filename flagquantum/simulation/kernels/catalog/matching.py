"""Policy-neutral matching over declared kernel implementation capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .evidence import EVIDENCE
from .implementations import IMPLEMENTATIONS
from .schema import (
    KERNEL_DEVICES,
    KernelDirection,
    KernelEvidence,
    KernelImplementation,
    KernelMaturity,
    KernelProvider,
    KernelSemantic,
)
from .semantics import SEMANTICS

KernelMismatchCode = Literal[
    "provider",
    "device",
    "dtype",
    "layout",
    "direction",
    "addressing",
    "maturity",
    "evidence",
]

_MATURITY_RANK: dict[KernelMaturity, int] = {
    "experimental": 0,
    "provisional": 1,
    "stable": 2,
}


@dataclass(frozen=True, slots=True)
class KernelRequest:
    """Capabilities required for one provider-neutral kernel semantic.

    ``device`` must name a declared execution device. A device outside
    ``KERNEL_DEVICES`` is a malformed request and is refused here rather than
    reported as a capability mismatch, so callers cannot read a typo as an
    unsupported device. The annotation stays :class:`str` because callers derive
    it from ``torch.device.type``; the runtime check is the authority.
    """

    semantic_id: str
    device: str
    dtype: str
    layout: str
    direction: KernelDirection
    addressing: tuple[str, ...] = ()
    providers: tuple[KernelProvider, ...] = ()
    minimum_maturity: KernelMaturity | None = None

    def __post_init__(self) -> None:
        required = (self.semantic_id, self.device, self.dtype, self.layout)
        if any(not value for value in required):
            raise ValueError(
                "kernel request identity and capabilities must be non-empty"
            )
        if self.device not in KERNEL_DEVICES:
            raise ValueError(
                f"undeclared kernel request device: {self.device}; "
                f"declared devices are {', '.join(sorted(KERNEL_DEVICES))}"
            )
        if len(set(self.addressing)) != len(self.addressing):
            raise ValueError("kernel request addressing entries must be unique")
        if any(not value for value in self.addressing):
            raise ValueError("kernel request addressing entries must be non-empty")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("kernel request providers must be unique")


@dataclass(frozen=True, slots=True)
class KernelMismatch:
    """One declared capability that did not satisfy a request."""

    code: KernelMismatchCode
    requested: tuple[str, ...]
    available: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KernelCandidate:
    """One implementation that satisfies a request and has evidence."""

    implementation: KernelImplementation
    evidence: KernelEvidence


@dataclass(frozen=True, slots=True)
class KernelRejection:
    """One implementation and every reason it did not satisfy a request."""

    implementation: KernelImplementation
    mismatches: tuple[KernelMismatch, ...]


@dataclass(frozen=True, slots=True)
class KernelMatchResult:
    """All capability matches and rejections for one semantic request."""

    request: KernelRequest
    semantic_known: bool
    candidates: tuple[KernelCandidate, ...]
    rejections: tuple[KernelRejection, ...]

    @property
    def matched(self) -> bool:
        """Return whether at least one evidenced implementation matched."""

        return bool(self.candidates)


def _mismatch(
    code: KernelMismatchCode,
    requested: str | tuple[str, ...],
    available: str | tuple[str, ...],
) -> KernelMismatch:
    requested_values = (requested,) if isinstance(requested, str) else requested
    available_values = (available,) if isinstance(available, str) else available
    return KernelMismatch(
        code=code,
        requested=requested_values,
        available=available_values,
    )


def _capability_mismatches(
    request: KernelRequest,
    implementation: KernelImplementation,
    evidence: KernelEvidence | None,
) -> tuple[KernelMismatch, ...]:
    mismatches: list[KernelMismatch] = []
    if request.providers and implementation.provider not in request.providers:
        mismatches.append(
            _mismatch("provider", request.providers, implementation.provider)
        )
    if request.device not in implementation.devices:
        mismatches.append(_mismatch("device", request.device, implementation.devices))
    if request.dtype not in implementation.dtypes:
        mismatches.append(_mismatch("dtype", request.dtype, implementation.dtypes))
    if request.layout not in implementation.layouts:
        mismatches.append(_mismatch("layout", request.layout, implementation.layouts))
    if request.direction not in implementation.directions:
        mismatches.append(
            _mismatch("direction", request.direction, implementation.directions)
        )
    if not set(request.addressing).issubset(implementation.addressing):
        mismatches.append(
            _mismatch("addressing", request.addressing, implementation.addressing)
        )
    if (
        request.minimum_maturity is not None
        and _MATURITY_RANK[implementation.maturity]
        < _MATURITY_RANK[request.minimum_maturity]
    ):
        mismatches.append(
            _mismatch(
                "maturity",
                request.minimum_maturity,
                implementation.maturity,
            )
        )
    required_evidence = ["correctness"]
    if any(direction != "forward" for direction in implementation.directions):
        required_evidence.append("gradient")
    if implementation.internal_fallback:
        required_evidence.append("capability")
    available_evidence: list[str] = []
    if evidence is not None:
        if evidence.correctness_tests:
            available_evidence.append("correctness")
        if evidence.gradient_tests:
            available_evidence.append("gradient")
        if evidence.capability_tests:
            available_evidence.append("capability")
    if not set(required_evidence).issubset(available_evidence):
        mismatches.append(
            _mismatch(
                "evidence",
                tuple(required_evidence),
                tuple(available_evidence),
            )
        )
    return tuple(mismatches)


def match_kernel_implementations(
    request: KernelRequest,
    *,
    semantics: Sequence[KernelSemantic] = SEMANTICS,
    implementations: Sequence[KernelImplementation] = IMPLEMENTATIONS,
    evidence: Sequence[KernelEvidence] = EVIDENCE,
) -> KernelMatchResult:
    """Return every evidenced implementation compatible with ``request``.

    Results preserve catalog order. This function does not rank providers,
    import implementation modules, inspect hardware, or execute fallbacks.
    """

    evidence_by_implementation: dict[str, KernelEvidence] = {}
    for evidence_record in evidence:
        if evidence_record.implementation_id in evidence_by_implementation:
            raise ValueError(
                "duplicate evidence for implementation "
                f"{evidence_record.implementation_id}"
            )
        evidence_by_implementation[evidence_record.implementation_id] = evidence_record

    semantic_known = any(
        semantic.semantic_id == request.semantic_id for semantic in semantics
    )
    candidates: list[KernelCandidate] = []
    rejections: list[KernelRejection] = []
    for implementation in implementations:
        if implementation.semantic_id != request.semantic_id:
            continue
        record = evidence_by_implementation.get(implementation.implementation_id)
        mismatches = _capability_mismatches(request, implementation, record)
        if mismatches:
            rejections.append(
                KernelRejection(
                    implementation=implementation,
                    mismatches=mismatches,
                )
            )
        else:
            assert record is not None
            candidates.append(
                KernelCandidate(implementation=implementation, evidence=record)
            )
    return KernelMatchResult(
        request=request,
        semantic_known=semantic_known,
        candidates=tuple(candidates),
        rejections=tuple(rejections),
    )


__all__ = [
    "KernelCandidate",
    "KernelMatchResult",
    "KernelMismatch",
    "KernelMismatchCode",
    "KernelRejection",
    "KernelRequest",
    "match_kernel_implementations",
]
