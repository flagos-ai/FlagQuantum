"""Release-bound circuit assessment for validated regional Twin candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..core.ir import CircuitIR, ensure_circuit_ir
from .prediction import TwinPrediction

if TYPE_CHECKING:
    from .region_model import TwinRegionModel
    from .region_release import TwinRegionRelease

_ASSESSMENT_SCHEMA = "flagquantum.twin_release_assessment.v1"

TwinReleaseAssessmentStatus = Literal["released_exact_circuit", "outside_release"]
TwinRegionSupportAssessmentStatus = Literal[
    "released_exact_circuit",
    "within_envelope_unvalidated",
    "outside_envelope",
]

_RELEASED: TwinReleaseAssessmentStatus = "released_exact_circuit"
_OUTSIDE: TwinReleaseAssessmentStatus = "outside_release"
_SUPPORT_RELEASED: TwinRegionSupportAssessmentStatus = "released_exact_circuit"
_WITHIN_ENVELOPE: TwinRegionSupportAssessmentStatus = "within_envelope_unvalidated"
_OUTSIDE_ENVELOPE: TwinRegionSupportAssessmentStatus = "outside_envelope"
_SUPPORT_ASSESSMENT_SCHEMA = "flagquantum.twin_region_support_assessment.v1"

# Every reason token below is a stable, deterministic fact about why a circuit
# is outside a release. Structural tokens come from the regional coverage report
# so a release assessment never restates the region's own scope logic.
_CANDIDATE_REGION_MISMATCH = "candidate_region_identity_mismatch"
_SNAPSHOT_MISMATCH = "snapshot_identity_mismatch"
_TARGET_MISMATCH = "target_mismatch"
_MAPPING_MISMATCH = "physical_mapping_mismatch"
_CIRCUIT_OUTSIDE_RELEASE = "circuit_identity_outside_release"
_CIRCUIT_NOT_VALIDATED = "circuit_identity_not_validated"
_SUPPORT_ENVELOPE_MISMATCH = "support_envelope_mismatch"


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TwinReleaseAssessment:
    """Release-bound verdict for one exact circuit on one regional Twin.

    The verdict is available only as the result of
    :meth:`TwinRegionRelease.assess`. It carries a prediction only when the
    release identity, candidate regional model, snapshot, target, ordered
    physical mapping, structural coverage, and exact frozen circuit identity all
    match. It never carries a per-circuit confidence level or an error bound,
    and it does not authorize routing.
    """

    status: TwinReleaseAssessmentStatus
    release_identity: str
    circuit_identity: str
    physical_qubits: tuple[int, ...]
    prediction: TwinPrediction | None
    reasons: tuple[str, ...]
    schema: str = _ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ASSESSMENT_SCHEMA:
            raise ValueError("unsupported Twin release-assessment schema")
        if self.status not in {_RELEASED, _OUTSIDE}:
            raise ValueError("unsupported Twin release-assessment status")
        _digest(self.release_identity, "release_identity")
        _digest(self.circuit_identity, "circuit_identity")
        mapping = tuple(self.physical_qubits)
        if (
            not mapping
            or len(mapping) != len(set(mapping))
            or any(type(qubit) is not int or qubit < 0 for qubit in mapping)
        ):
            raise ValueError(
                "assessment physical_qubits must be unique and non-negative"
            )
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reasons))
        if any(not reason.strip() for reason in reasons):
            raise ValueError("assessment reasons must be non-empty strings")
        if self.status == _RELEASED:
            if self.prediction is None or reasons:
                raise ValueError(
                    "released_exact_circuit requires one prediction and no reasons"
                )
            if self.prediction.circuit_identity != self.circuit_identity:
                raise ValueError(
                    "released prediction does not match the assessed circuit"
                )
            if self.prediction.n_wires != len(mapping):
                raise ValueError(
                    "released prediction does not match the assessed mapping"
                )
        elif self.prediction is not None or not reasons:
            raise ValueError(
                "outside_release requires deterministic reasons and no prediction"
            )
        object.__setattr__(self, "physical_qubits", mapping)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class TwinRegionSupportAssessment:
    """Evidence-qualified support verdict for one regional Twin circuit.

    Exact released circuits carry a prediction. Structurally supported but
    unseen circuits and circuits outside the frozen support envelope never do.
    The result states no per-circuit confidence level or error bound and does
    not authorize routing.
    """

    status: TwinRegionSupportAssessmentStatus
    release_identity: str
    circuit_identity: str
    physical_qubits: tuple[int, ...]
    prediction: TwinPrediction | None
    reasons: tuple[str, ...]
    schema: str = _SUPPORT_ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUPPORT_ASSESSMENT_SCHEMA:
            raise ValueError("unsupported Twin region-support assessment schema")
        if self.status not in {
            _SUPPORT_RELEASED,
            _WITHIN_ENVELOPE,
            _OUTSIDE_ENVELOPE,
        }:
            raise ValueError("unsupported Twin region-support assessment status")
        _digest(self.release_identity, "release_identity")
        _digest(self.circuit_identity, "circuit_identity")
        mapping = tuple(self.physical_qubits)
        if (
            not mapping
            or len(mapping) != len(set(mapping))
            or any(type(qubit) is not int or qubit < 0 for qubit in mapping)
        ):
            raise ValueError(
                "assessment physical_qubits must be unique and non-negative"
            )
        reasons = tuple(dict.fromkeys(str(reason) for reason in self.reasons))
        if any(not reason.strip() for reason in reasons):
            raise ValueError("assessment reasons must be non-empty strings")
        if self.status == _SUPPORT_RELEASED:
            if self.prediction is None or reasons:
                raise ValueError(
                    "released_exact_circuit requires one prediction and no reasons"
                )
            if self.prediction.circuit_identity != self.circuit_identity:
                raise ValueError(
                    "released prediction does not match the assessed circuit"
                )
            if self.prediction.n_wires != len(mapping):
                raise ValueError(
                    "released prediction does not match the assessed mapping"
                )
        elif self.prediction is not None or not reasons:
            raise ValueError(
                f"{self.status} requires deterministic reasons and no prediction"
            )
        if self.status == _WITHIN_ENVELOPE and reasons != (_CIRCUIT_NOT_VALIDATED,):
            raise ValueError(
                "within_envelope_unvalidated requires only "
                "circuit_identity_not_validated"
            )
        object.__setattr__(self, "physical_qubits", mapping)
        object.__setattr__(self, "reasons", reasons)


def _release_context(
    release: TwinRegionRelease,
    region_twin: TwinRegionModel,
    circuit: Any,
    physical_qubits: Sequence[int],
) -> tuple[CircuitIR, tuple[int, ...], tuple[str, ...]]:
    """Return authoritative identities and structural blockers."""

    from .region_model import TwinRegionModel
    from .region_release import TwinRegionRelease

    if not isinstance(release, TwinRegionRelease):
        raise TypeError("release must be a TwinRegionRelease")
    if not isinstance(region_twin, TwinRegionModel):
        raise TypeError("region_twin must be a TwinRegionModel")
    ir = ensure_circuit_ir(circuit)
    mapping = tuple(int(qubit) for qubit in physical_qubits)

    reasons: list[str] = []
    if region_twin.identity != release.candidate_region_identity:
        reasons.append(_CANDIDATE_REGION_MISMATCH)
    if region_twin.twin.snapshot.identity != release.candidate_snapshot_identity:
        reasons.append(_SNAPSHOT_MISMATCH)
    if region_twin.target != release.target:
        reasons.append(_TARGET_MISMATCH)
    mapping_is_usable = (
        bool(mapping)
        and len(mapping) == ir.n_wires
        and len(mapping) == len(set(mapping))
        and all(qubit >= 0 for qubit in mapping)
    )
    if mapping != release.physical_qubits or not mapping_is_usable:
        reasons.append(_MAPPING_MISMATCH)
    if mapping_is_usable:
        coverage = region_twin.region.coverage_report(circuit, physical_qubits=mapping)
        reasons.extend(coverage.reasons)
    return ir, mapping, tuple(dict.fromkeys(reasons))


def assess_release(
    release: TwinRegionRelease,
    region_twin: TwinRegionModel,
    circuit: Any,
    *,
    physical_qubits: Sequence[int],
) -> TwinReleaseAssessment:
    """Assess one exact circuit against a released regional Twin.

    Every identity is computed internally. Callers never supply fingerprints.
    All mismatches are reported together so one assessment explains the complete
    refusal, and a prediction is produced only on a full match.
    """

    ir, mapping, structural_reasons = _release_context(
        release, region_twin, circuit, physical_qubits
    )
    reasons = list(structural_reasons)
    if ir.content_hash not in release.verified_circuit_identities:
        reasons.append(_CIRCUIT_OUTSIDE_RELEASE)

    outside = tuple(dict.fromkeys(reasons))
    release_identity = release.identity
    if outside:
        return TwinReleaseAssessment(
            status=_OUTSIDE,
            release_identity=release_identity,
            circuit_identity=ir.content_hash,
            physical_qubits=mapping,
            prediction=None,
            reasons=outside,
        )
    return TwinReleaseAssessment(
        status=_RELEASED,
        release_identity=release_identity,
        circuit_identity=ir.content_hash,
        physical_qubits=mapping,
        prediction=region_twin.predict(circuit, physical_qubits=mapping),
        reasons=(),
    )


def assess_region_support(
    release: TwinRegionRelease,
    region_twin: TwinRegionModel,
    circuit: Any,
    *,
    physical_qubits: Sequence[int],
) -> TwinRegionSupportAssessment:
    """Classify a circuit against a release's frozen regional envelope.

    The call is offline and read-only. It returns a prediction only for an
    exact circuit already verified by the release. Structural compatibility is
    never promoted into an accuracy, confidence, or routing claim.
    """

    ir, mapping, structural_blockers = _release_context(
        release, region_twin, circuit, physical_qubits
    )
    region = region_twin.region
    envelope_matches = (
        release.physical_qubits == region.physical_qubits
        and release.directed_couplers == region.directed_couplers
        and release.supported_operations == region.supported_operations
        and release.maximum_instruction_count == region.maximum_instruction_count
        and release.maximum_circuit_depth == region.maximum_circuit_depth
        and release.source_snapshot_identities == region.source_snapshot_identities
        and release.source_support_identities == region.source_support_identities
    )
    blockers = tuple(
        dict.fromkeys(
            (
                *structural_blockers,
                *((_SUPPORT_ENVELOPE_MISMATCH,) if not envelope_matches else ()),
            )
        )
    )
    release_identity = release.identity
    circuit_identity = ir.content_hash
    if blockers:
        return TwinRegionSupportAssessment(
            status=_OUTSIDE_ENVELOPE,
            release_identity=release_identity,
            circuit_identity=circuit_identity,
            physical_qubits=mapping,
            prediction=None,
            reasons=blockers,
        )
    if circuit_identity not in release.verified_circuit_identities:
        return TwinRegionSupportAssessment(
            status=_WITHIN_ENVELOPE,
            release_identity=release_identity,
            circuit_identity=circuit_identity,
            physical_qubits=mapping,
            prediction=None,
            reasons=(_CIRCUIT_NOT_VALIDATED,),
        )
    return TwinRegionSupportAssessment(
        status=_SUPPORT_RELEASED,
        release_identity=release_identity,
        circuit_identity=circuit_identity,
        physical_qubits=mapping,
        prediction=region_twin.predict(circuit, physical_qubits=mapping),
        reasons=(),
    )


__all__ = (
    "assess_region_support",
    "assess_release",
    "TwinRegionSupportAssessment",
    "TwinRegionSupportAssessmentStatus",
    "TwinReleaseAssessment",
    "TwinReleaseAssessmentStatus",
)
