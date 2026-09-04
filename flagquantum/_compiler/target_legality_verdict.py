"""Compiler-owned legality verdict for legacy target capability coverage.

The Core v1 projection is useful for sharing a conservative vocabulary, but it
does not replace the legacy Compiler comparator.  This module records the
single, auditable decision made by that comparator.  It intentionally has no
Runtime, platform, or execution-authority meaning.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from .capability_comparison import CapabilityComparison
from .target_capabilities import TargetCapabilities
from .target_capabilities_adapter import (
    CompilerProjectionLoss,
    CompilerRequirementProjection,
)

COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA = (
    "flagquantum.compiler.target_legality_verdict.v1"
)
COMPILER_TARGET_LEGALITY_SCOPE = "compiler_target_legality"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _loss_key(loss: CompilerProjectionLoss) -> tuple[object, ...]:
    return (
        loss.field,
        loss.code.value,
        loss.semantic_value_json,
        loss.active,
        loss.message,
        loss.authority,
    )


def _loss_payload(
    losses: tuple[CompilerProjectionLoss, ...],
) -> list[dict[str, object]]:
    return [loss.to_dict() for loss in sorted(losses, key=_loss_key)]


def _comparison_payload(comparison: CapabilityComparison) -> dict[str, object]:
    return {
        "compatible": comparison.compatible,
        # Keep the comparator's established order; it is part of diagnostics.
        "differences": [item.to_dict() for item in comparison.differences],
        "diagnostics": [item.to_dict() for item in comparison.diagnostics],
    }


def _payload(verdict: "CompilerLegalityVerdict") -> dict[str, object]:
    return {
        "schema": verdict.schema,
        "scope": verdict.scope,
        "verdict": verdict.verdict,
        "required_legacy_semantic_fingerprint": verdict.required_legacy_semantic_fingerprint,
        "available_legacy_semantic_fingerprint": verdict.available_legacy_semantic_fingerprint,
        "requirement_set_id": verdict.requirement_set_id,
        "loss_accounting_identity": verdict.loss_accounting_identity,
        "requires_legacy_comparator": verdict.requires_legacy_comparator,
        "legacy_comparison": _comparison_payload(verdict.legacy_comparison),
        "losses": _loss_payload(verdict.losses),
    }


@dataclass(frozen=True)
class CompilerLegalityVerdict:
    """Closed attestation of Compiler target legality.

    ``verdict`` means only that ``available`` satisfies the Compiler's target
    capability requirement.  It is not a Runtime execution decision, platform
    availability claim, hardware evidence, or execution authorization.
    """

    verdict: bool
    required_legacy_semantic_fingerprint: str
    available_legacy_semantic_fingerprint: str
    requirement_set_id: str
    loss_accounting_identity: str
    requires_legacy_comparator: bool
    legacy_comparison: CapabilityComparison
    losses: tuple[CompilerProjectionLoss, ...]
    identity: str = ""
    schema: str = COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA
    scope: str = COMPILER_TARGET_LEGALITY_SCOPE

    def __post_init__(self) -> None:
        if self.schema != COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA:
            raise ValueError("unsupported Compiler legality verdict schema")
        if self.scope != COMPILER_TARGET_LEGALITY_SCOPE:
            raise ValueError("invalid Compiler legality verdict scope")
        if (
            type(self.verdict) is not bool
            or type(self.requires_legacy_comparator) is not bool
        ):
            raise ValueError("Compiler legality verdict flags must be boolean")
        for name in (
            "required_legacy_semantic_fingerprint",
            "available_legacy_semantic_fingerprint",
            "requirement_set_id",
            "loss_accounting_identity",
            "identity",
        ):
            value = getattr(self, name)
            if name == "identity" and value == "":
                continue
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 identity")
        if not isinstance(self.legacy_comparison, CapabilityComparison):
            raise TypeError("legacy_comparison must be a CapabilityComparison")
        if not isinstance(self.losses, tuple) or not all(
            isinstance(loss, CompilerProjectionLoss) for loss in self.losses
        ):
            raise TypeError("losses must be a tuple of CompilerProjectionLoss values")
        if self.verdict != self.legacy_comparison.compatible:
            raise ValueError("verdict must preserve legacy comparator compatibility")
        expected_loss_identity = _sha256(_loss_payload(self.losses))
        if self.loss_accounting_identity != expected_loss_identity:
            raise ValueError("loss accounting identity does not match losses")
        if self.identity:
            if not self.identity_valid:
                raise ValueError("Compiler legality verdict identity is invalid")
        else:
            object.__setattr__(self, "identity", _sha256(_payload(self)))

    @property
    def legacy_comparison_compatible(self) -> bool:
        return self.legacy_comparison.compatible

    @property
    def issue_paths(self) -> tuple[str, ...]:
        return tuple(item.field for item in self.legacy_comparison.differences)

    @property
    def identity_valid(self) -> bool:
        try:
            return _sha256(
                _loss_payload(self.losses)
            ) == self.loss_accounting_identity and (
                _sha256(_payload(self)) == self.identity
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

    def require_valid(self) -> "CompilerLegalityVerdict":
        """Fail closed if a serialized or in-memory attestation was tampered with."""

        if not self.identity_valid:
            raise ValueError("Compiler legality verdict identity is invalid")
        return self

    def to_dict(self) -> dict[str, object]:
        self.require_valid()
        return {**_payload(self), "identity": self.identity}


def evaluate_compiler_target_legality(
    projection: CompilerRequirementProjection,
    available: TargetCapabilities,
) -> CompilerLegalityVerdict:
    """Evaluate Compiler legality using the unchanged legacy comparator.

    No Core matcher result is consulted here.  Current projections always
    retain legacy comparator authority; a future comparator-exempt projection
    must introduce an explicit versioned rule before this function changes.
    """

    if not isinstance(projection, CompilerRequirementProjection):
        raise TypeError("projection must be a CompilerRequirementProjection")
    if not isinstance(available, TargetCapabilities):
        raise TypeError("available must be a Compiler TargetCapabilities value")
    comparison = projection.compare_available(available)
    losses = tuple(projection.losses)
    loss_identity = _sha256(_loss_payload(losses))
    # The legacy comparator remains authoritative even when Core can match the
    # projected subset.  This is also the only explicit rule for the current
    # (and comparator-required) projection schema.
    verdict = comparison.compatible
    provisional = CompilerLegalityVerdict(
        verdict=verdict,
        required_legacy_semantic_fingerprint=projection.legacy_semantic_fingerprint,
        available_legacy_semantic_fingerprint=available.semantic_fingerprint,
        requirement_set_id=projection.requirement_set.requirement_set_id,
        loss_accounting_identity=loss_identity,
        requires_legacy_comparator=projection.requires_legacy_comparator,
        legacy_comparison=comparison,
        losses=losses,
        identity="",
    )
    return replace(provisional, identity=_sha256(_payload(provisional)))


__all__ = [
    "COMPILER_TARGET_LEGALITY_SCOPE",
    "COMPILER_TARGET_LEGALITY_VERDICT_SCHEMA",
    "CompilerLegalityVerdict",
    "evaluate_compiler_target_legality",
]
