"""Bounded local-parameter estimates from frozen Twin probe observations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._statistics import finite_shot_tv_radius, total_variation
from .experiment import TwinHardwareReport
from .probe_bank import TwinProbeBank, TwinProbeBankStatus

_PARAMETER_ESTIMATE_SCHEMA = "flagquantum.twin_parameter_estimate.v1"
_ASSUMPTIONS = (
    "fixed_linear_local_error_burden_model",
    "independent_multinomial_hardware_reports",
    "simultaneous_bonferroni_finite_shot_bound",
    "parameters_bounded_to_unit_interval",
)


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


def _inverse(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    size = len(matrix)
    augmented = [
        [*row, *(1.0 if column == index else 0.0 for column in range(size))]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ValueError("probe design is numerically singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(tuple(row[size:]) for row in augmented)


def _left_inverse(design: tuple[tuple[int, ...], ...]) -> tuple[tuple[float, ...], ...]:
    row_count = len(design)
    parameter_count = len(design[0])
    gram = tuple(
        tuple(
            sum(
                float(design[row][left] * design[row][right])
                for row in range(row_count)
            )
            for right in range(parameter_count)
        )
        for left in range(parameter_count)
    )
    inverse = _inverse(gram)
    return tuple(
        tuple(
            sum(
                inverse[parameter][inner] * float(design[row][inner])
                for inner in range(parameter_count)
            )
            for row in range(row_count)
        )
        for parameter in range(parameter_count)
    )


def _full_probabilities(
    report: TwinHardwareReport, *, n_qubits: int
) -> tuple[tuple[float, ...], int]:
    bins = [0] * (2**n_qubits)
    if not isinstance(report.counts, Mapping):
        raise TypeError("hardware report counts must be a mapping")
    for raw_bitstring, raw_count in report.counts.items():
        bitstring = str(raw_bitstring)
        if len(bitstring) != n_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError("hardware count keys must be fixed-width bitstrings")
        if type(raw_count) is not int or raw_count < 0:
            raise ValueError("hardware counts must be non-negative integers")
        bins[int(bitstring, 2)] += raw_count
    shots = sum(bins)
    if shots <= 0 or shots != report.validation.shots:
        raise ValueError("hardware counts must match positive validation shots")
    probabilities = tuple(count / shots for count in bins)
    if len(report.validation.hardware_probabilities) != len(probabilities) or any(
        not math.isclose(observed, expected, abs_tol=1e-12)
        for observed, expected in zip(
            report.validation.hardware_probabilities, probabilities, strict=True
        )
    ):
        raise ValueError("hardware probabilities do not match report counts")
    return probabilities, shots


def _marginal(
    probabilities: Sequence[float],
    *,
    n_qubits: int,
    logical_qubits: tuple[int, ...],
) -> tuple[float, ...]:
    values = [0.0] * (2 ** len(logical_qubits))
    for basis_index, probability in enumerate(probabilities):
        bitstring = format(basis_index, f"0{n_qubits}b")
        outcome = "".join(bitstring[index] for index in logical_qubits)
        values[int(outcome, 2)] += float(probability)
    return tuple(values)


@dataclass(frozen=True)
class TwinParameterEstimate:
    """An immutable, assumption-qualified fit over one identifiable probe bank."""

    probe_bank_identity: str
    target: str
    physical_qubits: tuple[int, ...]
    parameter_family: str
    parameter_values: tuple[tuple[str, float], ...]
    confidence_intervals: tuple[tuple[str, float, float], ...]
    identifiability_status: TwinProbeBankStatus
    unidentified_parameters: tuple[str, ...]
    boundary_parameters: tuple[str, ...]
    hardware_report_identities: tuple[str, ...]
    task_ids: tuple[str, ...]
    snapshot_identity: str
    shots_by_probe: tuple[tuple[str, int], ...]
    observed_row_total_variations: tuple[float, ...]
    fitted_row_total_variations: tuple[float, ...]
    row_finite_shot_radii: tuple[float, ...]
    residual_root_mean_square: float
    maximum_absolute_residual: float
    goodness_of_fit: float
    confidence_level: float
    assumptions: tuple[str, ...] = _ASSUMPTIONS
    schema: str = _PARAMETER_ESTIMATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PARAMETER_ESTIMATE_SCHEMA:
            raise ValueError("unsupported Twin parameter-estimate schema")
        _digest(self.probe_bank_identity, "probe_bank_identity")
        _digest(self.snapshot_identity, "snapshot_identity")
        target_parts = self.target.split(":")
        if len(target_parts) != 2 or any(not part.strip() for part in target_parts):
            raise ValueError("target must use canonical provider:backend form")
        physical_qubits = tuple(self.physical_qubits)
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(type(qubit) is not int or qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique non-negative integers")
        if self.parameter_family != "local_pauli_readout":
            raise ValueError("unsupported parameter_family")
        values = tuple((name, float(value)) for name, value in self.parameter_values)
        names = tuple(name for name, _ in values)
        if (
            not names
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for _, value in values
            )
        ):
            raise ValueError("parameter values must be unique finite unit values")
        intervals = tuple(
            (name, float(lower), float(upper))
            for name, lower, upper in self.confidence_intervals
        )
        if tuple(name for name, _, _ in intervals) != names or any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 <= lower <= value <= upper <= 1.0
            for (_, value), (_, lower, upper) in zip(values, intervals, strict=True)
        ):
            raise ValueError("confidence intervals must bound every parameter value")
        if (
            self.identifiability_status != "identifiable"
            or self.unidentified_parameters
        ):
            raise ValueError("parameter estimates require a fully identifiable bank")
        boundaries = tuple(self.boundary_parameters)
        if len(boundaries) != len(set(boundaries)) or any(
            name not in names for name in boundaries
        ):
            raise ValueError("boundary_parameters must name unique fitted parameters")
        reports = tuple(self.hardware_report_identities)
        tasks = tuple(self.task_ids)
        if not reports or len(reports) != len(set(reports)):
            raise ValueError("hardware report identities must be non-empty and unique")
        if (
            not tasks
            or any(not isinstance(task, str) or not task for task in tasks)
            or len(tasks) != len(set(tasks))
            or len(tasks) != len(reports)
        ):
            raise ValueError("task IDs must uniquely align with hardware reports")
        for identity in reports:
            _digest(identity, "hardware_report_identity")
        shots = tuple(self.shots_by_probe)
        if (
            not shots
            or len(shots) != len(set(name for name, _ in shots))
            or any(value <= 0 for _, value in shots)
        ):
            raise ValueError("shots_by_probe must contain unique positive totals")
        for circuit_identity, _ in shots:
            _digest(circuit_identity, "probe_circuit_identity")
        observed = tuple(float(value) for value in self.observed_row_total_variations)
        fitted = tuple(float(value) for value in self.fitted_row_total_variations)
        radii = tuple(float(value) for value in self.row_finite_shot_radii)
        if (
            not observed
            or not (len(observed) == len(fitted) == len(radii))
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (*observed, *fitted, *radii)
            )
            or any(value > 1.0 for value in (*observed, *fitted))
        ):
            raise ValueError(
                "row observations, fits, and radii must align and be finite"
            )
        residuals = tuple(
            actual - predicted
            for actual, predicted in zip(observed, fitted, strict=True)
        )
        expected_rmse = math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        )
        expected_maximum = max(abs(value) for value in residuals)
        expected_goodness = max(0.0, 1.0 - expected_rmse)
        for name, actual, expected in (
            (
                "residual_root_mean_square",
                self.residual_root_mean_square,
                expected_rmse,
            ),
            (
                "maximum_absolute_residual",
                self.maximum_absolute_residual,
                expected_maximum,
            ),
            ("goodness_of_fit", self.goodness_of_fit, expected_goodness),
        ):
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, abs_tol=1e-12
            ):
                raise ValueError(f"{name} does not match the frozen fit")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        if tuple(self.assumptions) != _ASSUMPTIONS:
            raise ValueError("unsupported Twin parameter-estimation assumptions")
        object.__setattr__(self, "parameter_values", values)
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "confidence_intervals", intervals)
        object.__setattr__(self, "boundary_parameters", boundaries)
        object.__setattr__(self, "hardware_report_identities", reports)
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(self, "shots_by_probe", shots)
        object.__setattr__(self, "observed_row_total_variations", observed)
        object.__setattr__(self, "fitted_row_total_variations", fitted)
        object.__setattr__(self, "row_finite_shot_radii", radii)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of the complete frozen estimate."""

        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema": self.schema,
            "probe_bank_identity": self.probe_bank_identity,
            "target": self.target,
            "physical_qubits": list(self.physical_qubits),
            "parameter_family": self.parameter_family,
            "parameter_values": [
                {"name": name, "value": value} for name, value in self.parameter_values
            ],
            "confidence_intervals": [
                {"name": name, "lower": lower, "upper": upper}
                for name, lower, upper in self.confidence_intervals
            ],
            "identifiability_status": self.identifiability_status,
            "unidentified_parameters": list(self.unidentified_parameters),
            "boundary_parameters": list(self.boundary_parameters),
            "hardware_report_identities": list(self.hardware_report_identities),
            "task_ids": list(self.task_ids),
            "snapshot_identity": self.snapshot_identity,
            "shots_by_probe": [
                {"circuit_identity": identity, "shots": shots}
                for identity, shots in self.shots_by_probe
            ],
            "observed_row_total_variations": list(self.observed_row_total_variations),
            "fitted_row_total_variations": list(self.fitted_row_total_variations),
            "row_finite_shot_radii": list(self.row_finite_shot_radii),
            "residual_root_mean_square": self.residual_root_mean_square,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "goodness_of_fit": self.goodness_of_fit,
            "confidence_level": self.confidence_level,
            "assumptions": list(self.assumptions),
        }


def estimate_probe_parameters(
    bank: TwinProbeBank,
    hardware_reports: Sequence[TwinHardwareReport],
    *,
    confidence_level: float = 0.95,
) -> TwinParameterEstimate:
    """Fit bounded parameters from distinct, identity-matched probe reports."""

    if not isinstance(bank, TwinProbeBank):
        raise TypeError("bank must be a TwinProbeBank")
    if bank.identifiability_status != "identifiable":
        missing_text = ", ".join(bank.unidentified_parameters)
        raise ValueError(f"probe bank is underdetermined; unidentified: {missing_text}")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if isinstance(hardware_reports, (str, bytes)):
        raise TypeError("hardware_reports must be a sequence")
    reports = tuple(hardware_reports)
    if not reports or any(not isinstance(item, TwinHardwareReport) for item in reports):
        raise TypeError("hardware_reports must contain TwinHardwareReport objects")
    reports = tuple(sorted(reports, key=lambda item: item.identity))
    identities = tuple(report.identity for report in reports)
    task_ids = tuple(report.task_id for report in reports)
    if len(identities) != len(set(identities)):
        raise ValueError("hardware reports must be distinct and cannot be reused")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("hardware reports must identify distinct provider tasks")

    provider, backend = bank.target.split(":", 1)
    grouped: dict[str, list[tuple[tuple[float, ...], int]]] = {
        identity: [] for identity in bank.probe_circuit_identities
    }
    snapshot_identities: set[str] = set()
    for report in reports:
        if report.provider.lower() != provider or report.backend_name != backend:
            raise ValueError("hardware report target does not match the probe bank")
        mapping = report.result_metadata.get("target_qubits")
        if (
            not isinstance(mapping, (list, tuple))
            or tuple(mapping) != bank.physical_qubits
        ):
            raise ValueError(
                "hardware report physical mapping does not match the probe bank"
            )
        circuit_identity = report.validation.circuit_identity
        if circuit_identity not in grouped:
            raise ValueError("hardware report circuit is outside the probe bank")
        if report.executed_qasm_identity is None:
            raise ValueError(
                "hardware report requires an authoritative executed program"
            )
        _digest(report.executed_qasm_identity, "executed_qasm_identity")
        snapshot_identities.add(report.validation.snapshot_identity)
        grouped[circuit_identity].append(
            _full_probabilities(report, n_qubits=len(bank.physical_qubits))
        )
    if any(not values for values in grouped.values()):
        missing_probes = tuple(
            identity for identity, values in grouped.items() if not values
        )
        raise ValueError(
            f"hardware reports are missing probe circuits: {missing_probes}"
        )
    if len(snapshot_identities) != 1:
        raise ValueError("hardware reports must share one frozen Twin snapshot")
    snapshot_identity = next(iter(snapshot_identities))
    _digest(snapshot_identity, "snapshot_identity")

    pooled: dict[str, tuple[tuple[float, ...], int]] = {}
    for circuit_identity, source_observations in grouped.items():
        shots = sum(item[1] for item in source_observations)
        probabilities = tuple(
            sum(values[index] * count for values, count in source_observations) / shots
            for index in range(2 ** len(bank.physical_qubits))
        )
        pooled[circuit_identity] = (probabilities, shots)

    row_count = len(bank.design_matrix)
    per_row_confidence = 1.0 - (1.0 - confidence) / row_count
    rows_per_probe = len(bank.physical_qubits) + len(bank.directed_couplers)
    logical_indices = {
        physical: logical for logical, physical in enumerate(bank.physical_qubits)
    }
    row_observations: list[float] = []
    row_radii: list[float] = []
    for probe_index, circuit_identity in enumerate(bank.probe_circuit_identities):
        probabilities, shots = pooled[circuit_identity]
        for local_row in range(rows_per_probe):
            absolute_row = probe_index * rows_per_probe + local_row
            if local_row < len(bank.physical_qubits):
                logical_qubits: tuple[int, ...] = (local_row,)
            else:
                source, target = bank.directed_couplers[
                    local_row - len(bank.physical_qubits)
                ]
                logical_qubits = (
                    logical_indices[source],
                    logical_indices[target],
                )
            observed = _marginal(
                probabilities,
                n_qubits=len(bank.physical_qubits),
                logical_qubits=logical_qubits,
            )
            ideal = bank.ideal_row_probabilities[absolute_row]
            row_observations.append(total_variation(ideal, observed))
            row_radii.append(
                finite_shot_tv_radius(
                    outcome_count=len(ideal),
                    shots=shots,
                    confidence_level=per_row_confidence,
                )
            )

    inverse = _left_inverse(bank.design_matrix)
    raw_parameters = tuple(
        sum(weight * value for weight, value in zip(row, row_observations, strict=True))
        for row in inverse
    )
    fitted_parameters = tuple(min(1.0, max(0.0, value)) for value in raw_parameters)
    interval_radii = tuple(
        sum(abs(weight) * radius for weight, radius in zip(row, row_radii, strict=True))
        for row in inverse
    )
    intervals = tuple(
        (
            name,
            max(0.0, value - radius),
            min(1.0, value + radius),
        )
        for name, value, radius in zip(
            bank.parameter_names, fitted_parameters, interval_radii, strict=True
        )
    )
    boundary_parameters = tuple(
        name
        for (name, value), (_, lower, upper), raw in zip(
            zip(bank.parameter_names, fitted_parameters, strict=True),
            intervals,
            raw_parameters,
            strict=True,
        )
        if raw != value or value in (0.0, 1.0)
    )
    fitted_rows = tuple(
        sum(
            float(coefficient) * value
            for coefficient, value in zip(row, fitted_parameters, strict=True)
        )
        for row in bank.design_matrix
    )
    residuals = tuple(
        actual - predicted
        for actual, predicted in zip(row_observations, fitted_rows, strict=True)
    )
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    maximum = max(abs(value) for value in residuals)
    return TwinParameterEstimate(
        probe_bank_identity=bank.identity,
        target=bank.target,
        physical_qubits=bank.physical_qubits,
        parameter_family=bank.parameter_family,
        parameter_values=tuple(
            zip(bank.parameter_names, fitted_parameters, strict=True)
        ),
        confidence_intervals=intervals,
        identifiability_status=bank.identifiability_status,
        unidentified_parameters=bank.unidentified_parameters,
        boundary_parameters=boundary_parameters,
        hardware_report_identities=identities,
        task_ids=task_ids,
        snapshot_identity=snapshot_identity,
        shots_by_probe=tuple(
            (identity, pooled[identity][1])
            for identity in bank.probe_circuit_identities
        ),
        observed_row_total_variations=tuple(row_observations),
        fitted_row_total_variations=fitted_rows,
        row_finite_shot_radii=tuple(row_radii),
        residual_root_mean_square=rmse,
        maximum_absolute_residual=maximum,
        goodness_of_fit=max(0.0, 1.0 - rmse),
        confidence_level=confidence,
    )


__all__ = ("estimate_probe_parameters", "TwinParameterEstimate")
