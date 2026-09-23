"""Explicit, exact-circuit releases for validated regional Twin candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ._atomic import write_once
from .candidate import TwinCandidateDecision, _matches_canonical_payload
from .region_candidate_holdout import (
    TwinRegionCandidateHoldoutEvaluation,
    TwinRegionCandidateHoldoutStudy,
)
from .region_model import TwinRegionModel

if TYPE_CHECKING:
    from .release_assessment import TwinReleaseAssessment

_RELEASE_SCHEMA = "flagquantum.twin_region_release.v1"
_EXACT_CIRCUIT_SCOPE = "exact_circuits"


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


def _decision_from_interval(lower: float, upper: float) -> TwinCandidateDecision:
    if lower > 0.0:
        return "improved"
    if upper < 0.0:
        return "degraded"
    return "inconclusive"


@dataclass(frozen=True)
class TwinRegionRelease:
    """Immutable qualification of one regional candidate on exact circuits.

    This record does not authorize arbitrary circuits, automatic routing, or
    mutation of an application's active model.
    """

    incumbent_region_identity: str
    candidate_region_identity: str
    incumbent_snapshot_identity: str
    candidate_snapshot_identity: str
    candidate_captured_at: str
    study_identity: str
    evaluation_identity: str
    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    supported_operations: tuple[str, ...]
    maximum_instruction_count: int
    maximum_circuit_depth: int
    source_snapshot_identities: tuple[str, ...]
    source_support_identities: tuple[str, ...]
    reference_circuit_identities: tuple[str, ...]
    holdout_circuit_identities: tuple[str, ...]
    confidence_level: float
    reference_decision: TwinCandidateDecision
    holdout_decision: TwinCandidateDecision
    reference_candidate_improvement_lower_bound: float
    reference_candidate_improvement_upper_bound: float
    holdout_candidate_improvement_lower_bound: float
    holdout_candidate_improvement_upper_bound: float
    scope: str = _EXACT_CIRCUIT_SCOPE
    routing_authorized: bool = False
    schema: str = _RELEASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _RELEASE_SCHEMA:
            raise ValueError("unsupported Twin region-release schema")
        if self.scope != _EXACT_CIRCUIT_SCOPE:
            raise ValueError("Twin region releases cover exact circuits only")
        if self.routing_authorized is not False:
            raise ValueError("Twin region releases cannot authorize routing")
        for name in (
            "incumbent_region_identity",
            "candidate_region_identity",
            "incumbent_snapshot_identity",
            "candidate_snapshot_identity",
            "study_identity",
            "evaluation_identity",
        ):
            _digest(str(getattr(self, name)), name)
        if self.incumbent_region_identity == self.candidate_region_identity:
            raise ValueError("Twin region release requires a distinct candidate")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("Twin region release requires provider and backend")
        try:
            captured = datetime.fromisoformat(self.candidate_captured_at)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Twin region release requires a calibration capture"
            ) from error
        if captured.tzinfo is None:
            raise ValueError("Twin region release capture must include a timezone")
        if not self.physical_qubits or len(self.physical_qubits) != len(
            set(self.physical_qubits)
        ):
            raise ValueError("Twin region release requires unique physical qubits")
        if any(type(qubit) is not int or qubit < 0 for qubit in self.physical_qubits):
            raise ValueError("Twin region-release qubits must be non-negative integers")
        physical = set(self.physical_qubits)
        if len(self.directed_couplers) != len(set(self.directed_couplers)) or any(
            type(source) is not int
            or type(target) is not int
            or source == target
            or source not in physical
            or target not in physical
            for source, target in self.directed_couplers
        ):
            raise ValueError("Twin region-release couplers must match its qubits")
        if not self.supported_operations or len(self.supported_operations) != len(
            set(self.supported_operations)
        ):
            raise ValueError("Twin region release requires unique operations")
        if any(
            not isinstance(operation, str) or not operation.strip()
            for operation in self.supported_operations
        ):
            raise ValueError("Twin region-release operations must be non-empty strings")
        if (
            type(self.maximum_instruction_count) is not int
            or type(self.maximum_circuit_depth) is not int
            or self.maximum_instruction_count < 0
            or self.maximum_circuit_depth < 0
        ):
            raise ValueError("Twin region-release limits must be non-negative")
        if not self.source_snapshot_identities or len(
            self.source_snapshot_identities
        ) != len(self.source_support_identities):
            raise ValueError("Twin region release requires aligned source identities")
        for identity in (
            *self.source_snapshot_identities,
            *self.source_support_identities,
        ):
            _digest(identity, "source_identity")
        reference = tuple(self.reference_circuit_identities)
        holdout = tuple(self.holdout_circuit_identities)
        if len(reference) < 2 or len(holdout) < 2:
            raise ValueError("Twin region release requires both fixed circuit groups")
        if len(reference) != len(set(reference)) or len(holdout) != len(set(holdout)):
            raise ValueError("Twin region-release circuit identities must be unique")
        if set(reference) & set(holdout):
            raise ValueError("Twin region-release circuit groups must be disjoint")
        for identity in (*reference, *holdout):
            _digest(identity, "circuit_identity")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        bounds = (
            float(self.reference_candidate_improvement_lower_bound),
            float(self.reference_candidate_improvement_upper_bound),
            float(self.holdout_candidate_improvement_lower_bound),
            float(self.holdout_candidate_improvement_upper_bound),
        )
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in bounds
        ):
            raise ValueError("candidate-improvement bounds must be finite in [-1, 1]")
        reference_lower, reference_upper, holdout_lower, holdout_upper = bounds
        if reference_lower > reference_upper or holdout_lower > holdout_upper:
            raise ValueError("candidate-improvement intervals must be ordered")
        if self.reference_decision != _decision_from_interval(
            reference_lower, reference_upper
        ):
            raise ValueError("reference decision does not match its interval")
        if self.holdout_decision != _decision_from_interval(
            holdout_lower, holdout_upper
        ):
            raise ValueError("holdout decision does not match its interval")
        if self.reference_decision == "degraded" or self.holdout_decision != "improved":
            raise ValueError("Twin region release requires an improved holdout gate")
        object.__setattr__(self, "physical_qubits", tuple(self.physical_qubits))
        object.__setattr__(self, "directed_couplers", tuple(self.directed_couplers))
        object.__setattr__(
            self, "supported_operations", tuple(self.supported_operations)
        )
        object.__setattr__(
            self, "source_snapshot_identities", tuple(self.source_snapshot_identities)
        )
        object.__setattr__(
            self, "source_support_identities", tuple(self.source_support_identities)
        )
        object.__setattr__(self, "reference_circuit_identities", reference)
        object.__setattr__(self, "holdout_circuit_identities", holdout)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self, "reference_candidate_improvement_lower_bound", reference_lower
        )
        object.__setattr__(
            self, "reference_candidate_improvement_upper_bound", reference_upper
        )
        object.__setattr__(
            self, "holdout_candidate_improvement_lower_bound", holdout_lower
        )
        object.__setattr__(
            self, "holdout_candidate_improvement_upper_bound", holdout_upper
        )

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this bounded release."""

        return _identity(self.to_dict())

    @property
    def target(self) -> str:
        """Return the canonical provider/backend target."""

        return f"{self.provider}:{self.backend_name}"

    @property
    def verified_circuit_identities(self) -> tuple[str, ...]:
        """Return every exact circuit covered by this release."""

        return (*self.reference_circuit_identities, *self.holdout_circuit_identities)

    def assess(
        self,
        region_twin: TwinRegionModel,
        circuit: Any,
        *,
        physical_qubits: Sequence[int],
    ) -> TwinReleaseAssessment:
        """Assess one exact circuit against this release and one regional Twin.

        The verdict binds the release identity, candidate regional model,
        snapshot, target, ordered physical mapping, structural coverage, and the
        exact frozen circuit identity. A prediction is returned only when all of
        them match; every other case fails closed with deterministic reasons and
        no prediction. This method performs no provider I/O and never mutates
        the release or the model.

        ``region_twin`` is a composed regional model, for example from
        :func:`compose_region_twin` or the application's own regional-model
        loader. Its ordered mapping must equal the release mapping.

        Examples:
            release = fq.twin.load_region_release("region-release.json")
            assessment = release.assess(
                candidate_region_twin,
                fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
                physical_qubits=(20, 27, 34),
            )
            assert assessment.release_identity == release.identity
        """

        from .release_assessment import assess_release

        return assess_release(
            self, region_twin, circuit, physical_qubits=physical_qubits
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical version-1 release payload."""

        return {
            "schema": self.schema,
            "incumbent_region_identity": self.incumbent_region_identity,
            "candidate_region_identity": self.candidate_region_identity,
            "incumbent_snapshot_identity": self.incumbent_snapshot_identity,
            "candidate_snapshot_identity": self.candidate_snapshot_identity,
            "candidate_captured_at": self.candidate_captured_at,
            "study_identity": self.study_identity,
            "evaluation_identity": self.evaluation_identity,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "directed_couplers": [list(item) for item in self.directed_couplers],
            "supported_operations": list(self.supported_operations),
            "maximum_instruction_count": self.maximum_instruction_count,
            "maximum_circuit_depth": self.maximum_circuit_depth,
            "source_snapshot_identities": list(self.source_snapshot_identities),
            "source_support_identities": list(self.source_support_identities),
            "reference_circuit_identities": list(self.reference_circuit_identities),
            "holdout_circuit_identities": list(self.holdout_circuit_identities),
            "confidence_level": self.confidence_level,
            "reference_decision": self.reference_decision,
            "holdout_decision": self.holdout_decision,
            "reference_candidate_improvement_lower_bound": (
                self.reference_candidate_improvement_lower_bound
            ),
            "reference_candidate_improvement_upper_bound": (
                self.reference_candidate_improvement_upper_bound
            ),
            "holdout_candidate_improvement_lower_bound": (
                self.holdout_candidate_improvement_lower_bound
            ),
            "holdout_candidate_improvement_upper_bound": (
                self.holdout_candidate_improvement_upper_bound
            ),
            "scope": self.scope,
            "routing_authorized": self.routing_authorized,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinRegionRelease:
        """Restore a release from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin region release must be a mapping")
        expected = {
            "schema",
            "incumbent_region_identity",
            "candidate_region_identity",
            "incumbent_snapshot_identity",
            "candidate_snapshot_identity",
            "candidate_captured_at",
            "study_identity",
            "evaluation_identity",
            "provider",
            "backend_name",
            "physical_qubits",
            "directed_couplers",
            "supported_operations",
            "maximum_instruction_count",
            "maximum_circuit_depth",
            "source_snapshot_identities",
            "source_support_identities",
            "reference_circuit_identities",
            "holdout_circuit_identities",
            "confidence_level",
            "reference_decision",
            "holdout_decision",
            "reference_candidate_improvement_lower_bound",
            "reference_candidate_improvement_upper_bound",
            "holdout_candidate_improvement_lower_bound",
            "holdout_candidate_improvement_upper_bound",
            "scope",
            "routing_authorized",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "Twin region-release fields do not match the v1 schema: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        try:
            release = cls(
                schema=str(payload["schema"]),
                incumbent_region_identity=str(payload["incumbent_region_identity"]),
                candidate_region_identity=str(payload["candidate_region_identity"]),
                incumbent_snapshot_identity=str(payload["incumbent_snapshot_identity"]),
                candidate_snapshot_identity=str(payload["candidate_snapshot_identity"]),
                candidate_captured_at=str(payload["candidate_captured_at"]),
                study_identity=str(payload["study_identity"]),
                evaluation_identity=str(payload["evaluation_identity"]),
                provider=str(payload["provider"]),
                backend_name=str(payload["backend_name"]),
                physical_qubits=tuple(payload["physical_qubits"]),
                directed_couplers=tuple(
                    tuple(item) for item in payload["directed_couplers"]
                ),
                supported_operations=tuple(payload["supported_operations"]),
                maximum_instruction_count=int(payload["maximum_instruction_count"]),
                maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
                source_snapshot_identities=tuple(payload["source_snapshot_identities"]),
                source_support_identities=tuple(payload["source_support_identities"]),
                reference_circuit_identities=tuple(
                    payload["reference_circuit_identities"]
                ),
                holdout_circuit_identities=tuple(payload["holdout_circuit_identities"]),
                confidence_level=float(payload["confidence_level"]),
                reference_decision=cast(
                    TwinCandidateDecision, str(payload["reference_decision"])
                ),
                holdout_decision=cast(
                    TwinCandidateDecision, str(payload["holdout_decision"])
                ),
                reference_candidate_improvement_lower_bound=float(
                    payload["reference_candidate_improvement_lower_bound"]
                ),
                reference_candidate_improvement_upper_bound=float(
                    payload["reference_candidate_improvement_upper_bound"]
                ),
                holdout_candidate_improvement_lower_bound=float(
                    payload["holdout_candidate_improvement_lower_bound"]
                ),
                holdout_candidate_improvement_upper_bound=float(
                    payload["holdout_candidate_improvement_upper_bound"]
                ),
                scope=str(payload["scope"]),
                routing_authorized=payload["routing_authorized"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin region release") from error
        if not _matches_canonical_payload(release.to_dict(), payload):
            raise ValueError("Twin region release is not in canonical v1 form")
        return release


def release_region_candidate(
    incumbent: TwinRegionModel,
    candidate: TwinRegionModel,
    *,
    study: TwinRegionCandidateHoldoutStudy,
    evaluation: TwinRegionCandidateHoldoutEvaluation,
) -> TwinRegionRelease:
    """Create an exact-circuit release after an explicit improved gate.

    Examples:
        release = fq.twin.release_region_candidate(
            incumbent_region,
            candidate_region,
            study=study,
            evaluation=evaluation,
        )
        fq.twin.dump_region_release(release, "region-release.json")

    This function is offline. It returns an immutable qualification artifact;
    it does not mutate either model or authorize workload routing.
    """

    if not isinstance(incumbent, TwinRegionModel) or not isinstance(
        candidate, TwinRegionModel
    ):
        raise TypeError("incumbent and candidate must be TwinRegionModel objects")
    if not isinstance(study, TwinRegionCandidateHoldoutStudy):
        raise TypeError("study must be a TwinRegionCandidateHoldoutStudy")
    if not isinstance(evaluation, TwinRegionCandidateHoldoutEvaluation):
        raise TypeError("evaluation must be a TwinRegionCandidateHoldoutEvaluation")
    if (incumbent.identity, candidate.identity) != (
        study.incumbent_region_identity,
        study.candidate_region_identity,
    ):
        raise ValueError("regional models do not match the frozen candidate study")
    if evaluation.study_identity != study.identity:
        raise ValueError("evaluation does not match the frozen candidate study")
    if evaluation.reference_evaluation.suite_identity != study.reference_suite.identity:
        raise ValueError("reference evaluation does not match the frozen suite")
    if evaluation.holdout_evaluation.suite_identity != study.holdout_suite.identity:
        raise ValueError("holdout evaluation does not match the frozen suite")
    if evaluation.decision != "improved":
        raise ValueError("regional candidate release requires an improved decision")
    incumbent_snapshot = incumbent.twin.snapshot.identity
    candidate_snapshot = candidate.twin.snapshot.identity
    for item in (
        *evaluation.reference_evaluation.evaluations,
        *evaluation.holdout_evaluation.evaluations,
    ):
        if (
            item.incumbent_snapshot_identity,
            item.candidate_snapshot_identity,
            item.hardware_report.provider,
            item.hardware_report.backend_name,
        ) != (
            incumbent_snapshot,
            candidate_snapshot,
            candidate.region.provider,
            candidate.region.backend_name,
        ):
            raise ValueError("candidate evaluation does not match the regional models")
    reference = tuple(
        trial.incumbent_prediction.circuit_identity
        for trial in study.reference_suite.trials
    )
    holdout = tuple(
        trial.incumbent_prediction.circuit_identity
        for trial in study.holdout_suite.trials
    )
    region = candidate.region
    return TwinRegionRelease(
        incumbent_region_identity=incumbent.identity,
        candidate_region_identity=candidate.identity,
        incumbent_snapshot_identity=incumbent_snapshot,
        candidate_snapshot_identity=candidate_snapshot,
        candidate_captured_at=candidate.twin.snapshot.captured_at,
        study_identity=study.identity,
        evaluation_identity=evaluation.identity,
        provider=region.provider,
        backend_name=region.backend_name,
        physical_qubits=region.physical_qubits,
        directed_couplers=region.directed_couplers,
        supported_operations=region.supported_operations,
        maximum_instruction_count=region.maximum_instruction_count,
        maximum_circuit_depth=region.maximum_circuit_depth,
        source_snapshot_identities=region.source_snapshot_identities,
        source_support_identities=region.source_support_identities,
        reference_circuit_identities=reference,
        holdout_circuit_identities=holdout,
        confidence_level=evaluation.confidence_level,
        reference_decision=evaluation.reference_evaluation.decision,
        holdout_decision=evaluation.holdout_evaluation.decision,
        reference_candidate_improvement_lower_bound=(
            evaluation.reference_evaluation.candidate_improvement_lower_bound
        ),
        reference_candidate_improvement_upper_bound=(
            evaluation.reference_evaluation.candidate_improvement_upper_bound
        ),
        holdout_candidate_improvement_lower_bound=(
            evaluation.holdout_evaluation.candidate_improvement_lower_bound
        ),
        holdout_candidate_improvement_upper_bound=(
            evaluation.holdout_evaluation.candidate_improvement_upper_bound
        ),
    )


def load_region_release(path: str | PathLike[str]) -> TwinRegionRelease:
    """Load a bounded release without provider contact or model mutation."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin region release from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin region-release file must contain a JSON object")
    return TwinRegionRelease.from_dict(payload)


def dump_region_release(
    release: TwinRegionRelease,
    path: str | PathLike[str],
) -> None:
    """Write one private release without replacing different content."""

    if not isinstance(release, TwinRegionRelease):
        raise TypeError("release must be a TwinRegionRelease")
    encoded = (
        json.dumps(
            release.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin region release",
        matches=lambda destination: load_region_release(destination) == release,
    )


__all__ = (
    "dump_region_release",
    "load_region_release",
    "release_region_candidate",
    "TwinRegionRelease",
)
