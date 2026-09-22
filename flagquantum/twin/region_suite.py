"""Prospective validation suites for connected regional Twin models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import fmean
from typing import Any

from ..remote.qpu import DeploymentResult
from ._atomic import write_once
from .circuit_support import TwinCircuitSupport
from .evidence import TwinEvidenceEnvelope
from .experiment import TwinExperiment
from .region_model import TwinRegionModel
from .series import TwinValidationSeries
from .submission import TwinSubmission, _experiment_from_dict, _require_fields

_SUITE_SCHEMA = "flagquantum.twin_region_validation_suite.v1"
_EVALUATION_SCHEMA = "flagquantum.twin_region_suite_evaluation.v1"
_EVALUATION_FIELDS = {
    "schema",
    "suite_identity",
    "region_identity",
    "snapshot_identity",
    "physical_qubits",
    "validation_series",
    "directed_couplers",
    "maximum_circuit_depth",
    "confidence_level",
    "circuit_count",
    "task_count",
    "total_shots",
    "mean_twin_qpu_agreement",
    "mean_ideal_qpu_agreement",
    "mean_qpu_repeatability",
    "simultaneous_finite_shot_tv_radius",
    "simultaneous_tv_error_bound",
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


@dataclass(frozen=True)
class TwinRegionSuiteEvaluation:
    """Simultaneous repeated evidence for a fixed regional circuit suite."""

    suite_identity: str
    region_identity: str
    snapshot_identity: str
    physical_qubits: tuple[int, ...]
    validation_series: tuple[TwinValidationSeries, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    maximum_circuit_depth: int
    confidence_level: float
    schema: str = _EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _EVALUATION_SCHEMA:
            raise ValueError("unsupported Twin region-suite evaluation schema")
        _digest(self.suite_identity, "suite_identity")
        _digest(self.region_identity, "region_identity")
        _digest(self.snapshot_identity, "snapshot_identity")
        qubits = tuple(self.physical_qubits)
        if (
            not qubits
            or len(qubits) != len(set(qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
        ):
            raise ValueError(
                "region-suite evaluation physical qubits must be unique "
                "non-negative integers"
            )
        series = tuple(self.validation_series)
        if len(series) < 2 or any(
            not isinstance(item, TwinValidationSeries) for item in series
        ):
            raise ValueError(
                "region-suite evaluation requires at least two validation series"
            )
        if any(item.repetitions < 2 for item in series):
            raise ValueError(
                "region-suite evaluation requires repeated tasks for every circuit"
            )
        if any(item.snapshot_identity != self.snapshot_identity for item in series):
            raise ValueError("region-suite series must use one Twin snapshot")
        if any(item.physical_qubits != qubits for item in series):
            raise ValueError("region-suite series must use one physical mapping")
        circuit_identities = tuple(item.circuit_identity for item in series)
        if len(circuit_identities) != len(set(circuit_identities)):
            raise ValueError("region-suite series must cover distinct circuits")
        report_identities = tuple(
            identity for item in series for identity in item.report_identities
        )
        if len(report_identities) != len(set(report_identities)):
            raise ValueError("region-suite evaluation requires distinct QPU tasks")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        per_circuit_confidence = 1.0 - (1.0 - confidence) / len(series)
        if any(
            not math.isclose(
                item.confidence_level, per_circuit_confidence, abs_tol=1e-12
            )
            for item in series
        ):
            raise ValueError(
                "region-suite validation series require simultaneous confidence"
            )
        couplers = tuple(self.directed_couplers)
        physical = set(qubits)
        if len(couplers) != len(set(couplers)) or any(
            source == target or source not in physical or target not in physical
            for source, target in couplers
        ):
            raise ValueError(
                "region-suite couplers must be unique directed mapped interactions"
            )
        maximum_depth = int(self.maximum_circuit_depth)
        if maximum_depth < 0:
            raise ValueError("maximum_circuit_depth must be non-negative")
        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "validation_series", series)
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "maximum_circuit_depth", maximum_depth)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def circuit_count(self) -> int:
        return len(self.validation_series)

    @property
    def task_count(self) -> int:
        return sum(item.repetitions for item in self.validation_series)

    @property
    def total_shots(self) -> int:
        return sum(item.total_shots for item in self.validation_series)

    @property
    def mean_twin_qpu_agreement(self) -> float:
        return fmean(item.mean_twin_qpu_agreement for item in self.validation_series)

    @property
    def mean_ideal_qpu_agreement(self) -> float:
        return fmean(item.mean_ideal_qpu_agreement for item in self.validation_series)

    @property
    def mean_qpu_repeatability(self) -> float:
        repeatability: list[float] = []
        for item in self.validation_series:
            value = item.mean_qpu_repeatability
            if value is None:
                raise RuntimeError("region-suite repeatability is unavailable")
            repeatability.append(value)
        return fmean(repeatability)

    @property
    def simultaneous_finite_shot_tv_radius(self) -> float:
        return max(
            item.simultaneous_finite_shot_tv_radius for item in self.validation_series
        )

    @property
    def simultaneous_tv_error_bound(self) -> float:
        return max(item.verified_tv_error_bound for item in self.validation_series)

    def to_circuit_support(self) -> TwinCircuitSupport:
        """Return exact-circuit support for the complete validated suite."""

        operations = tuple(
            sorted(
                {
                    operation
                    for item in self.validation_series
                    for operation in item.supported_operations
                }
            )
        )
        evidence = TwinEvidenceEnvelope(
            snapshot_identity=self.snapshot_identity,
            physical_qubits=self.physical_qubits,
            supported_operations=operations,
            maximum_instruction_count=max(
                item.maximum_instruction_count for item in self.validation_series
            ),
            verified_circuit_identities=tuple(
                item.circuit_identity for item in self.validation_series
            ),
            evidence_identity=self.identity,
            verified_tv_error_bound=self.simultaneous_tv_error_bound,
            estimated_tv_error_bound=None,
            confidence_level=self.confidence_level,
        )
        return TwinCircuitSupport(
            evidence=evidence,
            directed_couplers=self.directed_couplers,
            maximum_circuit_depth=self.maximum_circuit_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "suite_identity": self.suite_identity,
            "region_identity": self.region_identity,
            "snapshot_identity": self.snapshot_identity,
            "physical_qubits": list(self.physical_qubits),
            "validation_series": [item.to_dict() for item in self.validation_series],
            "directed_couplers": [list(item) for item in self.directed_couplers],
            "maximum_circuit_depth": self.maximum_circuit_depth,
            "confidence_level": self.confidence_level,
            "circuit_count": self.circuit_count,
            "task_count": self.task_count,
            "total_shots": self.total_shots,
            "mean_twin_qpu_agreement": self.mean_twin_qpu_agreement,
            "mean_ideal_qpu_agreement": self.mean_ideal_qpu_agreement,
            "mean_qpu_repeatability": self.mean_qpu_repeatability,
            "simultaneous_finite_shot_tv_radius": (
                self.simultaneous_finite_shot_tv_radius
            ),
            "simultaneous_tv_error_bound": self.simultaneous_tv_error_bound,
        }


def _evaluation_from_dict(
    payload: Mapping[str, Any],
) -> TwinRegionSuiteEvaluation:
    """Restore one suite evaluation for trusted package persistence code."""

    if not isinstance(payload, Mapping):
        raise TypeError("Twin region-suite evaluation must be a mapping")
    actual = set(payload)
    if actual != _EVALUATION_FIELDS:
        raise ValueError(
            "Twin region-suite evaluation fields do not match the v1 schema: "
            f"missing={sorted(_EVALUATION_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _EVALUATION_FIELDS)}"
        )
    series_payloads = payload["validation_series"]
    if not isinstance(series_payloads, list) or any(
        not isinstance(item, Mapping) for item in series_payloads
    ):
        raise ValueError("Twin region-suite series must be JSON objects")
    evaluation = TwinRegionSuiteEvaluation(
        schema=str(payload["schema"]),
        suite_identity=str(payload["suite_identity"]),
        region_identity=str(payload["region_identity"]),
        snapshot_identity=str(payload["snapshot_identity"]),
        physical_qubits=tuple(payload["physical_qubits"]),
        validation_series=tuple(
            TwinValidationSeries.from_dict(item) for item in series_payloads
        ),
        directed_couplers=tuple(tuple(item) for item in payload["directed_couplers"]),
        maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
        confidence_level=float(payload["confidence_level"]),
    )
    if _identity(evaluation.to_dict()) != _identity(payload):
        raise ValueError("Twin region-suite evaluation is not in canonical v1 form")
    return evaluation


@dataclass(frozen=True)
class TwinRegionValidationSuite:
    """A fixed regional circuit suite with repeated-task requirements."""

    region_identity: str
    snapshot_identity: str
    physical_qubits: tuple[int, ...]
    experiments: tuple[TwinExperiment, ...]
    repetitions: int
    directed_couplers: tuple[tuple[int, int], ...]
    maximum_circuit_depth: int
    schema: str = _SUITE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SUITE_SCHEMA:
            raise ValueError("unsupported Twin region-validation suite schema")
        _digest(self.region_identity, "region_identity")
        _digest(self.snapshot_identity, "snapshot_identity")
        qubits = tuple(self.physical_qubits)
        if (
            not qubits
            or len(qubits) != len(set(qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in qubits)
        ):
            raise ValueError(
                "region-validation physical qubits must be unique non-negative integers"
            )
        experiments = tuple(self.experiments)
        if len(experiments) < 2 or any(
            not isinstance(item, TwinExperiment) for item in experiments
        ):
            raise ValueError(
                "region-validation suite requires at least two experiments"
            )
        if any(
            item.snapshot_identity != self.snapshot_identity for item in experiments
        ):
            raise ValueError("region-validation experiments must use one Twin snapshot")
        if any(item.target_qubits != qubits for item in experiments):
            raise ValueError("region-validation experiments must use one mapping")
        circuit_identities = tuple(
            item.prediction.circuit_identity for item in experiments
        )
        names = tuple(item.name for item in experiments)
        if len(circuit_identities) != len(set(circuit_identities)):
            raise ValueError("region-validation suite requires distinct circuits")
        if len(names) != len(set(names)):
            raise ValueError("region-validation suite requires distinct names")
        repetitions = int(self.repetitions)
        if repetitions < 2:
            raise ValueError(
                "region-validation suite requires at least two repetitions"
            )
        couplers = tuple(self.directed_couplers)
        physical = set(qubits)
        if len(couplers) != len(set(couplers)) or any(
            source == target or source not in physical or target not in physical
            for source, target in couplers
        ):
            raise ValueError(
                "region-validation couplers must be unique directed mapped interactions"
            )
        maximum_depth = int(self.maximum_circuit_depth)
        if maximum_depth < 0:
            raise ValueError("maximum_circuit_depth must be non-negative")
        object.__setattr__(self, "physical_qubits", qubits)
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "maximum_circuit_depth", maximum_depth)

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def circuit_count(self) -> int:
        return len(self.experiments)

    @property
    def planned_task_count(self) -> int:
        return self.circuit_count * self.repetitions

    @property
    def planned_shots(self) -> int:
        return sum(item.shots for item in self.experiments) * self.repetitions

    def validate_results(
        self,
        submissions: Sequence[TwinSubmission],
        results: Sequence[DeploymentResult],
        *,
        circuits: Sequence[Any],
        confidence_level: float = 0.95,
    ) -> TwinRegionSuiteEvaluation:
        """Validate every frozen task without submitting, polling, or retrying."""

        frozen_submissions = tuple(submissions)
        frozen_results = tuple(results)
        frozen_circuits = tuple(circuits)
        if len(frozen_circuits) != self.circuit_count:
            raise ValueError("circuits must match every suite experiment")
        if not (
            len(frozen_submissions) == len(frozen_results) == self.planned_task_count
        ):
            raise ValueError(
                "submissions and results must match every planned suite task"
            )
        confidence = float(confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        if any(not isinstance(item, TwinSubmission) for item in frozen_submissions):
            raise TypeError("submissions must contain only TwinSubmission objects")
        task_ids = tuple(item.receipt.task_id for item in frozen_submissions)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("region-validation suite requires distinct QPU tasks")
        per_circuit_confidence = 1.0 - (1.0 - confidence) / self.circuit_count
        series: list[TwinValidationSeries] = []
        offset = 0
        for experiment, circuit in zip(self.experiments, frozen_circuits, strict=True):
            reports = []
            for _ in range(self.repetitions):
                submission = frozen_submissions[offset]
                result = frozen_results[offset]
                if submission.experiment != experiment:
                    raise ValueError(
                        "submission does not match its suite experiment and repetition"
                    )
                reports.append(submission.validate_result(result))
                offset += 1
            series.append(
                experiment.validation_series(
                    reports,
                    circuit=circuit,
                    confidence_level=per_circuit_confidence,
                )
            )
        return TwinRegionSuiteEvaluation(
            suite_identity=self.identity,
            region_identity=self.region_identity,
            snapshot_identity=self.snapshot_identity,
            physical_qubits=self.physical_qubits,
            validation_series=tuple(series),
            directed_couplers=self.directed_couplers,
            maximum_circuit_depth=self.maximum_circuit_depth,
            confidence_level=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "region_identity": self.region_identity,
            "snapshot_identity": self.snapshot_identity,
            "physical_qubits": list(self.physical_qubits),
            "experiments": [item.to_dict() for item in self.experiments],
            "repetitions": self.repetitions,
            "directed_couplers": [list(item) for item in self.directed_couplers],
            "maximum_circuit_depth": self.maximum_circuit_depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinRegionValidationSuite:
        """Restore a suite from its strict version-1 serialized form."""

        if not isinstance(payload, Mapping):
            raise TypeError("Twin region-validation suite must be a mapping")
        _require_fields(
            payload,
            {
                "schema",
                "region_identity",
                "snapshot_identity",
                "physical_qubits",
                "experiments",
                "repetitions",
                "directed_couplers",
                "maximum_circuit_depth",
            },
            name="Twin region-validation suite",
        )
        experiments = payload["experiments"]
        if not isinstance(experiments, list) or any(
            not isinstance(item, Mapping) for item in experiments
        ):
            raise ValueError("Twin region-validation experiments must be JSON objects")
        try:
            return cls(
                schema=str(payload["schema"]),
                region_identity=str(payload["region_identity"]),
                snapshot_identity=str(payload["snapshot_identity"]),
                physical_qubits=tuple(payload["physical_qubits"]),
                experiments=tuple(_experiment_from_dict(item) for item in experiments),
                repetitions=int(payload["repetitions"]),
                directed_couplers=tuple(
                    tuple(item) for item in payload["directed_couplers"]
                ),
                maximum_circuit_depth=int(payload["maximum_circuit_depth"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Twin region-validation suite") from error


def prepare_region_validation_suite(
    region_twin: TwinRegionModel,
    circuits: Sequence[Any],
    *,
    physical_qubits: Sequence[int],
    name: str,
    shots: int,
    repetitions: int = 2,
) -> TwinRegionValidationSuite:
    """Freeze distinct covered circuits without submitting any QPU task."""

    if not isinstance(region_twin, TwinRegionModel):
        raise TypeError("region_twin must be a TwinRegionModel")
    frozen_circuits = tuple(circuits)
    if len(frozen_circuits) < 2:
        raise ValueError("region-validation suite requires at least two circuits")
    mapping = tuple(int(qubit) for qubit in physical_qubits)
    base_name = str(name).strip()
    if not base_name:
        raise ValueError("name must be non-empty")
    experiments: list[TwinExperiment] = []
    couplers: list[tuple[int, int]] = []
    depths: list[int] = []
    for index, circuit in enumerate(frozen_circuits, start=1):
        coverage = region_twin.region.coverage_report(circuit, physical_qubits=mapping)
        experiments.append(
            region_twin.prepare_experiment(
                circuit,
                physical_qubits=mapping,
                name=f"{base_name}-{index:02d}",
                shots=shots,
            )
        )
        couplers.extend(coverage.required_directed_couplers)
        depths.append(coverage.circuit_depth)
    return TwinRegionValidationSuite(
        region_identity=region_twin.identity,
        snapshot_identity=region_twin.twin.snapshot.identity,
        physical_qubits=mapping,
        experiments=tuple(experiments),
        repetitions=repetitions,
        directed_couplers=tuple(dict.fromkeys(couplers)),
        maximum_circuit_depth=max(depths, default=0),
    )


def load_region_validation_suite(
    path: str | PathLike[str],
) -> TwinRegionValidationSuite:
    """Load a frozen regional suite without contacting a provider."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin region-validation suite from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin region-validation suite file must contain a JSON object")
    return TwinRegionValidationSuite.from_dict(payload)


def dump_region_validation_suite(
    suite: TwinRegionValidationSuite,
    path: str | PathLike[str],
) -> None:
    """Write a private, create-once regional validation-suite file."""

    if not isinstance(suite, TwinRegionValidationSuite):
        raise TypeError("suite must be a TwinRegionValidationSuite")
    encoded = (
        json.dumps(
            suite.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    write_once(
        Path(path),
        encoded,
        label="Twin region-validation suite",
        matches=lambda destination: (
            load_region_validation_suite(destination).identity == suite.identity
        ),
    )


__all__ = (
    "dump_region_validation_suite",
    "load_region_validation_suite",
    "prepare_region_validation_suite",
    "TwinRegionSuiteEvaluation",
    "TwinRegionValidationSuite",
)
