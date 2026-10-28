"""Immutable operator decisions for completed Twin evolution cycles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, cast

from ._atomic import write_once
from .evolution_cycle import TwinEvolutionCycle, TwinEvolutionRecommendation
from .region_release import TwinRegionRelease

_SCHEMA = "flagquantum.twin_evolution_decision.v1"
_DECISIONS = {
    "retain_incumbent",
    "collect_more_evidence",
    "release_candidate",
}


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: str | None, name: str) -> str | None:
    return None if value is None else _digest(value, name)


@dataclass(frozen=True)
class TwinEvolutionDecision:
    """One explicit, audit-ready decision about an evolution cycle.

    The record preserves the evidence recommendation separately from the
    operator decision. It does not release or activate a model, authorize
    routing, or perform provider operations.
    """

    cycle_identity: str
    recommended_action: TwinEvolutionRecommendation
    decision: TwinEvolutionRecommendation
    decided_at: str
    decision_authority: str
    target: str
    physical_qubits: tuple[int, ...]
    incumbent_snapshot_identity: str
    candidate_snapshot_identity: str | None
    holdout_evaluation_identity: str | None
    release_identity: str | None = None
    released_region_identity: str | None = None
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported Twin evolution-decision schema")
        for name in ("cycle_identity", "incumbent_snapshot_identity"):
            _digest(str(getattr(self, name)), name)
        for name in (
            "candidate_snapshot_identity",
            "holdout_evaluation_identity",
            "release_identity",
            "released_region_identity",
        ):
            _optional_digest(getattr(self, name), name)
        if self.recommended_action not in _DECISIONS or self.decision not in _DECISIONS:
            raise ValueError("unsupported Twin evolution decision")
        try:
            decided = datetime.fromisoformat(self.decided_at)
        except (TypeError, ValueError) as error:
            raise ValueError("decided_at must be an ISO-8601 timestamp") from error
        if decided.tzinfo is None:
            raise ValueError("decided_at must include a timezone")
        if not self.decision_authority.strip():
            raise ValueError("decision_authority must be non-empty")
        if not self.target.strip() or ":" not in self.target:
            raise ValueError("target must use provider:backend form")
        physical_qubits = tuple(self.physical_qubits)
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique non-negative integers")
        release_fields = (self.release_identity, self.released_region_identity)
        if self.decision == "release_candidate":
            if self.recommended_action != "release_candidate":
                raise ValueError(
                    "a candidate cannot be released against the evidence gate"
                )
            if self.candidate_snapshot_identity is None:
                raise ValueError("candidate release requires a candidate snapshot")
            if self.holdout_evaluation_identity is None:
                raise ValueError("candidate release requires a holdout evaluation")
            if not all(release_fields):
                raise ValueError("candidate release requires an explicit release")
        elif any(release_fields):
            raise ValueError("non-release decisions cannot bind a release")
        object.__setattr__(self, "physical_qubits", physical_qubits)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this decision record."""

        return _identity(self.to_dict())

    @property
    def followed_recommendation(self) -> bool:
        """Report whether the operator selected the evidence recommendation."""

        return self.decision == self.recommended_action

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical version-1 decision payload."""

        return {
            "schema": self.schema,
            "cycle_identity": self.cycle_identity,
            "recommended_action": self.recommended_action,
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decision_authority": self.decision_authority,
            "target": self.target,
            "physical_qubits": list(self.physical_qubits),
            "incumbent_snapshot_identity": self.incumbent_snapshot_identity,
            "candidate_snapshot_identity": self.candidate_snapshot_identity,
            "holdout_evaluation_identity": self.holdout_evaluation_identity,
            "release_identity": self.release_identity,
            "released_region_identity": self.released_region_identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinEvolutionDecision:
        """Restore a strict version-1 decision record."""

        expected = {
            "schema",
            "cycle_identity",
            "recommended_action",
            "decision",
            "decided_at",
            "decision_authority",
            "target",
            "physical_qubits",
            "incumbent_snapshot_identity",
            "candidate_snapshot_identity",
            "holdout_evaluation_identity",
            "release_identity",
            "released_region_identity",
        }
        if set(payload) != expected:
            raise ValueError(
                "Twin evolution-decision fields do not match the v1 schema"
            )
        try:
            decision = cls(
                schema=str(payload["schema"]),
                cycle_identity=str(payload["cycle_identity"]),
                recommended_action=cast(
                    TwinEvolutionRecommendation, payload["recommended_action"]
                ),
                decision=cast(TwinEvolutionRecommendation, payload["decision"]),
                decided_at=str(payload["decided_at"]),
                decision_authority=str(payload["decision_authority"]),
                target=str(payload["target"]),
                physical_qubits=tuple(int(item) for item in payload["physical_qubits"]),
                incumbent_snapshot_identity=str(payload["incumbent_snapshot_identity"]),
                candidate_snapshot_identity=_optional_text(
                    payload["candidate_snapshot_identity"]
                ),
                holdout_evaluation_identity=_optional_text(
                    payload["holdout_evaluation_identity"]
                ),
                release_identity=_optional_text(payload["release_identity"]),
                released_region_identity=_optional_text(
                    payload["released_region_identity"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid Twin evolution decision") from error
        if decision.to_dict() != dict(payload):
            raise ValueError("Twin evolution decision is not in canonical v1 form")
        return decision


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def record_evolution_decision(
    cycle: TwinEvolutionCycle,
    *,
    decision: TwinEvolutionRecommendation,
    decided_at: str,
    decision_authority: str,
    release: TwinRegionRelease | None = None,
) -> TwinEvolutionDecision:
    """Record an operator decision without applying it.

    A release decision requires an already-created ``TwinRegionRelease`` bound
    to the candidate snapshot, target, and physical mapping. Retain and collect
    decisions must not carry a release.

    Examples:
        record = fq.twin.record_evolution_decision(
            cycle,
            decision="collect_more_evidence",
            decided_at="2026-08-15T11:00:00+08:00",
            decision_authority="calibration-operator",
        )
    """

    if not isinstance(cycle, TwinEvolutionCycle):
        raise TypeError("cycle must be a TwinEvolutionCycle")
    if release is not None and not isinstance(release, TwinRegionRelease):
        raise TypeError("release must be a TwinRegionRelease")
    if decision == "release_candidate":
        if cycle.stage != "holdout_evaluated":
            raise ValueError("candidate release requires a completed holdout gate")
        if release is None:
            raise ValueError("candidate release requires an explicit release")
        if (
            release.candidate_snapshot_identity != cycle.candidate_snapshot_identity
            or release.target != cycle.target
            or release.physical_qubits != cycle.physical_qubits
        ):
            raise ValueError("release does not match the evolution cycle")
    elif release is not None:
        raise ValueError("non-release decisions cannot bind a release")
    return TwinEvolutionDecision(
        cycle_identity=cycle.identity,
        recommended_action=cycle.recommended_action,
        decision=decision,
        decided_at=decided_at,
        decision_authority=decision_authority,
        target=cycle.target,
        physical_qubits=cycle.physical_qubits,
        incumbent_snapshot_identity=cycle.incumbent_snapshot_identity,
        candidate_snapshot_identity=cycle.candidate_snapshot_identity,
        holdout_evaluation_identity=cycle.holdout_evaluation_identity,
        release_identity=None if release is None else release.identity,
        released_region_identity=(
            None if release is None else release.candidate_region_identity
        ),
    )


def dump_evolution_decision(
    decision: TwinEvolutionDecision, path: str | PathLike[str]
) -> None:
    """Save one decision once without replacing another artifact."""

    if not isinstance(decision, TwinEvolutionDecision):
        raise TypeError("decision must be a TwinEvolutionDecision")
    destination = Path(path)
    encoded = json.dumps(decision.to_dict(), sort_keys=True, indent=2) + "\n"
    write_once(
        destination,
        encoded,
        label="Twin evolution decision",
        matches=lambda existing: (
            load_evolution_decision(existing).identity == decision.identity
        ),
    )


def load_evolution_decision(path: str | PathLike[str]) -> TwinEvolutionDecision:
    """Load and strictly validate a saved decision record."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin evolution decision from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin evolution-decision file must contain a JSON object")
    return TwinEvolutionDecision.from_dict(payload)


__all__ = (
    "dump_evolution_decision",
    "load_evolution_decision",
    "record_evolution_decision",
    "TwinEvolutionDecision",
)
