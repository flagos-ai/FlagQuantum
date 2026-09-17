"""Conservative summaries of repeated QPU Twin validation results."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from os import PathLike
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core.ir import ensure_circuit_ir
from ._statistics import finite_shot_tv_radius, total_variation
from .evidence import TwinEvidenceEnvelope
from .experiment import TwinExperiment, TwinHardwareReport

_SERIES_SCHEMA = "flagquantum.twin_validation_series.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _unit_interval(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


@dataclass(frozen=True)
class TwinValidationSeries:
    """Evidence-qualified summary of distinct runs of one frozen experiment.

    Examples:
        series = experiment.validation_series(
            [first_report, second_report],
            circuit=circuit,
        )
        evidence = series.to_evidence()
        print(series.mean_twin_qpu_agreement)
        print(series.mean_qpu_repeatability)
    """

    snapshot_identity: str
    circuit_identity: str
    physical_qubits: tuple[int, ...]
    report_identities: tuple[str, ...]
    repetitions: int
    total_shots: int
    mean_twin_hardware_total_variation: float
    maximum_twin_hardware_total_variation: float
    mean_ideal_hardware_total_variation: float
    maximum_ideal_hardware_total_variation: float
    mean_hardware_repeatability_total_variation: float | None
    maximum_hardware_repeatability_total_variation: float | None
    simultaneous_finite_shot_tv_radius: float
    verified_tv_error_bound: float
    confidence_level: float
    supported_operations: tuple[str, ...]
    maximum_instruction_count: int
    schema: str = _SERIES_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SERIES_SCHEMA:
            raise ValueError("unsupported Twin validation-series schema")
        _digest(self.snapshot_identity, "snapshot_identity")
        _digest(self.circuit_identity, "circuit_identity")
        qubits = tuple(int(qubit) for qubit in self.physical_qubits)
        if (
            not qubits
            or len(qubits) != len(set(qubits))
            or any(qubit < 0 for qubit in qubits)
        ):
            raise ValueError(
                "physical_qubits must be non-empty, unique, and non-negative"
            )
        identities = tuple(
            _digest(value, "report_identities") for value in self.report_identities
        )
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("report_identities must be non-empty and unique")
        if type(self.repetitions) is not int or self.repetitions != len(identities):
            raise ValueError("repetitions must match report_identities")
        if (
            type(self.total_shots) is not int
            or self.total_shots < self.repetitions
            or self.total_shots % self.repetitions
        ):
            raise ValueError("total_shots must cover every repetition")

        mean_twin = _unit_interval(
            self.mean_twin_hardware_total_variation,
            "mean_twin_hardware_total_variation",
        )
        maximum_twin = _unit_interval(
            self.maximum_twin_hardware_total_variation,
            "maximum_twin_hardware_total_variation",
        )
        mean_ideal = _unit_interval(
            self.mean_ideal_hardware_total_variation,
            "mean_ideal_hardware_total_variation",
        )
        maximum_ideal = _unit_interval(
            self.maximum_ideal_hardware_total_variation,
            "maximum_ideal_hardware_total_variation",
        )
        if mean_twin > maximum_twin or mean_ideal > maximum_ideal:
            raise ValueError("mean validation distance cannot exceed its maximum")

        pairwise = (
            self.mean_hardware_repeatability_total_variation,
            self.maximum_hardware_repeatability_total_variation,
        )
        if self.repetitions == 1:
            if pairwise != (None, None):
                raise ValueError("one repetition has no hardware repeatability pair")
        elif pairwise[0] is None or pairwise[1] is None:
            raise ValueError("repeated validation requires hardware repeatability")
        else:
            mean_repeatability = _unit_interval(
                pairwise[0], "mean_hardware_repeatability_total_variation"
            )
            maximum_repeatability = _unit_interval(
                pairwise[1], "maximum_hardware_repeatability_total_variation"
            )
            if mean_repeatability > maximum_repeatability:
                raise ValueError(
                    "mean repeatability distance cannot exceed its maximum"
                )

        radius = _unit_interval(
            self.simultaneous_finite_shot_tv_radius,
            "simultaneous_finite_shot_tv_radius",
        )
        bound = _unit_interval(self.verified_tv_error_bound, "verified_tv_error_bound")
        expected_bound = min(1.0, maximum_twin + radius)
        if not math.isclose(bound, expected_bound, abs_tol=1e-12):
            raise ValueError("verified bound must equal observation plus shot radius")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        operations = tuple(
            str(operation).strip().lower() for operation in self.supported_operations
        )
        if not operations or any(not operation for operation in operations):
            raise ValueError("supported_operations must contain non-empty names")
        if len(operations) != len(set(operations)):
            raise ValueError("supported_operations must be unique")
        if self.maximum_instruction_count < 0:
            raise ValueError("maximum_instruction_count must be non-negative")

        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "report_identities", identities)
        object.__setattr__(self, "mean_twin_hardware_total_variation", mean_twin)
        object.__setattr__(self, "maximum_twin_hardware_total_variation", maximum_twin)
        object.__setattr__(self, "mean_ideal_hardware_total_variation", mean_ideal)
        object.__setattr__(
            self, "maximum_ideal_hardware_total_variation", maximum_ideal
        )
        object.__setattr__(
            self,
            "mean_hardware_repeatability_total_variation",
            None if pairwise[0] is None else float(pairwise[0]),
        )
        object.__setattr__(
            self,
            "maximum_hardware_repeatability_total_variation",
            None if pairwise[1] is None else float(pairwise[1]),
        )
        object.__setattr__(self, "simultaneous_finite_shot_tv_radius", radius)
        object.__setattr__(self, "verified_tv_error_bound", bound)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "supported_operations", operations)

    @classmethod
    def _from_reports(
        cls,
        experiment: TwinExperiment,
        reports: Sequence[TwinHardwareReport],
        *,
        circuit: Any,
        confidence_level: float = 0.95,
    ) -> TwinValidationSeries:
        """Summarize distinct, identity-bound results without assuming stationarity."""

        if not isinstance(experiment, TwinExperiment):
            raise TypeError("experiment must be a TwinExperiment")
        observations = tuple(reports)
        if not observations:
            raise ValueError("validation series requires at least one report")
        if any(not isinstance(report, TwinHardwareReport) for report in observations):
            raise TypeError("reports must contain only TwinHardwareReport objects")
        report_identities = tuple(report.identity for report in observations)
        task_ids = tuple(report.task_id for report in observations)
        if len(report_identities) != len(set(report_identities)) or len(
            task_ids
        ) != len(set(task_ids)):
            raise ValueError("validation series requires distinct hardware tasks")

        confidence = float(confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        repetitions = len(observations)
        per_report_confidence = 1.0 - (1.0 - confidence) / repetitions
        for report in observations:
            experiment.evidence_from_report(
                report,
                circuit=circuit,
                confidence_level=per_report_confidence,
            )

        validations = tuple(report.validation for report in observations)
        twin_distances = tuple(
            validation.twin_hardware_total_variation for validation in validations
        )
        ideal_distances = tuple(
            validation.ideal_hardware_total_variation for validation in validations
        )
        radii = tuple(
            finite_shot_tv_radius(
                outcome_count=2**experiment.prediction.n_wires,
                shots=validation.shots,
                confidence_level=per_report_confidence,
            )
            for validation in validations
        )
        repeatability = tuple(
            total_variation(left.hardware_probabilities, right.hardware_probabilities)
            for left, right in combinations(validations, 2)
        )
        ir = ensure_circuit_ir(circuit)
        operations = tuple(
            sorted({instruction.name for instruction in ir.instructions} | {"measure"})
        )
        return cls(
            snapshot_identity=experiment.snapshot_identity,
            circuit_identity=experiment.prediction.circuit_identity,
            physical_qubits=experiment.target_qubits,
            report_identities=report_identities,
            repetitions=repetitions,
            total_shots=sum(validation.shots for validation in validations),
            mean_twin_hardware_total_variation=fmean(twin_distances),
            maximum_twin_hardware_total_variation=max(twin_distances),
            mean_ideal_hardware_total_variation=fmean(ideal_distances),
            maximum_ideal_hardware_total_variation=max(ideal_distances),
            mean_hardware_repeatability_total_variation=(
                None if not repeatability else fmean(repeatability)
            ),
            maximum_hardware_repeatability_total_variation=(
                None if not repeatability else max(repeatability)
            ),
            simultaneous_finite_shot_tv_radius=max(radii),
            verified_tv_error_bound=max(
                min(1.0, distance + radius)
                for distance, radius in zip(twin_distances, radii, strict=True)
            ),
            confidence_level=confidence,
            supported_operations=operations,
            maximum_instruction_count=len(ir.instructions),
        )

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def mean_twin_qpu_agreement(self) -> float:
        """Return one minus mean Twin-to-hardware TV distance."""

        return 1.0 - self.mean_twin_hardware_total_variation

    @property
    def mean_ideal_qpu_agreement(self) -> float:
        """Return one minus mean ideal-to-hardware TV distance."""

        return 1.0 - self.mean_ideal_hardware_total_variation

    @property
    def mean_qpu_repeatability(self) -> float | None:
        """Return one minus mean pairwise hardware TV, when repetitions exist."""

        distance = self.mean_hardware_repeatability_total_variation
        return None if distance is None else 1.0 - distance

    def to_evidence(self) -> TwinEvidenceEnvelope:
        """Create exact-circuit evidence bound to this complete series."""

        return TwinEvidenceEnvelope(
            snapshot_identity=self.snapshot_identity,
            physical_qubits=self.physical_qubits,
            supported_operations=self.supported_operations,
            maximum_instruction_count=self.maximum_instruction_count,
            verified_circuit_identities=(self.circuit_identity,),
            evidence_identity=self.identity,
            verified_tv_error_bound=self.verified_tv_error_bound,
            estimated_tv_error_bound=None,
            confidence_level=self.confidence_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "circuit_identity": self.circuit_identity,
            "physical_qubits": list(self.physical_qubits),
            "report_identities": list(self.report_identities),
            "repetitions": self.repetitions,
            "total_shots": self.total_shots,
            "mean_twin_hardware_total_variation": (
                self.mean_twin_hardware_total_variation
            ),
            "maximum_twin_hardware_total_variation": (
                self.maximum_twin_hardware_total_variation
            ),
            "mean_ideal_hardware_total_variation": (
                self.mean_ideal_hardware_total_variation
            ),
            "maximum_ideal_hardware_total_variation": (
                self.maximum_ideal_hardware_total_variation
            ),
            "mean_hardware_repeatability_total_variation": (
                self.mean_hardware_repeatability_total_variation
            ),
            "maximum_hardware_repeatability_total_variation": (
                self.maximum_hardware_repeatability_total_variation
            ),
            "simultaneous_finite_shot_tv_radius": (
                self.simultaneous_finite_shot_tv_radius
            ),
            "verified_tv_error_bound": self.verified_tv_error_bound,
            "confidence_level": self.confidence_level,
            "supported_operations": list(self.supported_operations),
            "maximum_instruction_count": self.maximum_instruction_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinValidationSeries:
        """Restore a validation series from its strict versioned schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin validation series must be a mapping")
        expected = {
            "schema",
            "snapshot_identity",
            "circuit_identity",
            "physical_qubits",
            "report_identities",
            "repetitions",
            "total_shots",
            "mean_twin_hardware_total_variation",
            "maximum_twin_hardware_total_variation",
            "mean_ideal_hardware_total_variation",
            "maximum_ideal_hardware_total_variation",
            "mean_hardware_repeatability_total_variation",
            "maximum_hardware_repeatability_total_variation",
            "simultaneous_finite_shot_tv_radius",
            "verified_tv_error_bound",
            "confidence_level",
            "supported_operations",
            "maximum_instruction_count",
        }
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "Twin validation-series fields do not match the v1 schema: "
                f"missing={missing}, unexpected={unexpected}"
            )
        try:
            return cls(
                schema=str(payload["schema"]),
                snapshot_identity=str(payload["snapshot_identity"]),
                circuit_identity=str(payload["circuit_identity"]),
                physical_qubits=tuple(payload["physical_qubits"]),
                report_identities=tuple(payload["report_identities"]),
                repetitions=int(payload["repetitions"]),
                total_shots=int(payload["total_shots"]),
                mean_twin_hardware_total_variation=float(
                    payload["mean_twin_hardware_total_variation"]
                ),
                maximum_twin_hardware_total_variation=float(
                    payload["maximum_twin_hardware_total_variation"]
                ),
                mean_ideal_hardware_total_variation=float(
                    payload["mean_ideal_hardware_total_variation"]
                ),
                maximum_ideal_hardware_total_variation=float(
                    payload["maximum_ideal_hardware_total_variation"]
                ),
                mean_hardware_repeatability_total_variation=(
                    None
                    if payload["mean_hardware_repeatability_total_variation"] is None
                    else float(payload["mean_hardware_repeatability_total_variation"])
                ),
                maximum_hardware_repeatability_total_variation=(
                    None
                    if payload["maximum_hardware_repeatability_total_variation"] is None
                    else float(
                        payload["maximum_hardware_repeatability_total_variation"]
                    )
                ),
                simultaneous_finite_shot_tv_radius=float(
                    payload["simultaneous_finite_shot_tv_radius"]
                ),
                verified_tv_error_bound=float(payload["verified_tv_error_bound"]),
                confidence_level=float(payload["confidence_level"]),
                supported_operations=tuple(payload["supported_operations"]),
                maximum_instruction_count=int(payload["maximum_instruction_count"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin validation series") from error


def load_validation_series(path: str | PathLike[str]) -> TwinValidationSeries:
    """Load one validation series without contacting a provider.

    Examples:
        series = fq.twin.load_validation_series("twin-validation.json")

    Raises:
        ValueError: If the file is not a canonical v1 validation series.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Twin validation series from {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin validation-series file must contain a JSON object")
    return TwinValidationSeries.from_dict(payload)


def dump_validation_series(
    series: TwinValidationSeries,
    path: str | PathLike[str],
) -> None:
    """Write one canonical series without replacing different observations.

    Examples:
        fq.twin.dump_validation_series(series, "twin-validation.json")

    Raises:
        TypeError: If ``series`` is not a Twin validation series.
        ValueError: If the destination cannot be written safely.
    """

    if not isinstance(series, TwinValidationSeries):
        raise TypeError("series must be a TwinValidationSeries")
    destination = Path(path)
    encoded = (
        json.dumps(
            series.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_validation_series(destination)
        except ValueError as error:
            raise ValueError(
                "Refusing to replace invalid Twin validation series at "
                f"{destination}"
            ) from error
        if existing.identity != series.identity:
            raise ValueError(
                "Refusing to replace different Twin validation series at "
                f"{destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin validation series to {destination}"
        ) from error


__all__ = (
    "TwinValidationSeries",
    "dump_validation_series",
    "load_validation_series",
)
