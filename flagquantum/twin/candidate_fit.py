"""Build immutable QPU Twin candidates from bounded probe estimates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from ..core.operator_schema import get_operator_schema
from ..noise import (
    DeviceNoiseProfile,
    NoiseModel,
    QubitNoiseCalibration,
    ReadoutError,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from .model import QPUDigitalTwin, TwinSnapshot
from .parameter_estimation import TwinParameterEstimate

_FIT_SCHEMA = "flagquantum.twin_candidate_fit.v1"
_SOURCE_MARKER = "|flagquantum-candidate-fit:"
_READOUT_PATTERN = re.compile(r"^readout_bit_flip\[Q(\d+)\]$")
_SINGLE_PATTERN = re.compile(r"^single_qubit_pauli\[Q(\d+)\]$")
_TWO_PATTERN = re.compile(r"^two_qubit_pauli\[Q(\d+)->Q(\d+)\]$")


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
class TwinCandidateFit:
    """Provenance for one offline candidate constructed from a bounded fit."""

    incumbent_snapshot_identity: str
    parameter_estimate_identity: str
    probe_bank_identity: str
    parameter_family: str
    fitted_parameters: tuple[tuple[str, float, float, float], ...]
    retained_parameters: tuple[str, ...]
    unidentified_parameters: tuple[str, ...]
    schema: str = _FIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _FIT_SCHEMA:
            raise ValueError("unsupported Twin candidate-fit schema")
        _digest(self.incumbent_snapshot_identity, "incumbent_snapshot_identity")
        _digest(self.parameter_estimate_identity, "parameter_estimate_identity")
        _digest(self.probe_bank_identity, "probe_bank_identity")
        if self.parameter_family != "local_pauli_readout":
            raise ValueError("unsupported candidate-fit parameter family")
        fitted = tuple(
            (name, float(value), float(lower), float(upper))
            for name, value, lower, upper in self.fitted_parameters
        )
        fitted_names = tuple(name for name, *_ in fitted)
        if (
            not fitted
            or len(fitted_names) != len(set(fitted_names))
            or any(
                not name
                or not all(math.isfinite(item) for item in (value, lower, upper))
                or not 0.0 <= lower <= value <= upper <= 1.0
                for name, value, lower, upper in fitted
            )
        ):
            raise ValueError("fitted parameters must be unique bounded intervals")
        retained = tuple(self.retained_parameters)
        unidentified = tuple(self.unidentified_parameters)
        if (
            len(retained) != len(set(retained))
            or len(unidentified) != len(set(unidentified))
            or any(not name for name in (*retained, *unidentified))
            or set(fitted_names).intersection(retained)
            or set(fitted_names).intersection(unidentified)
            or set(retained).intersection(unidentified)
        ):
            raise ValueError("candidate parameter dispositions must be disjoint")
        object.__setattr__(self, "fitted_parameters", fitted)
        object.__setattr__(self, "retained_parameters", retained)
        object.__setattr__(self, "unidentified_parameters", unidentified)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of this complete fit provenance."""

        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema": self.schema,
            "incumbent_snapshot_identity": self.incumbent_snapshot_identity,
            "parameter_estimate_identity": self.parameter_estimate_identity,
            "probe_bank_identity": self.probe_bank_identity,
            "parameter_family": self.parameter_family,
            "fitted_parameters": [
                {
                    "name": name,
                    "value": value,
                    "lower": lower,
                    "upper": upper,
                }
                for name, value, lower, upper in self.fitted_parameters
            ],
            "retained_parameters": list(self.retained_parameters),
            "unidentified_parameters": list(self.unidentified_parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TwinCandidateFit:
        """Restore strict candidate-fit provenance from its canonical mapping."""

        expected = {
            "schema",
            "incumbent_snapshot_identity",
            "parameter_estimate_identity",
            "probe_bank_identity",
            "parameter_family",
            "fitted_parameters",
            "retained_parameters",
            "unidentified_parameters",
        }
        if set(payload) != expected:
            raise ValueError("Twin candidate-fit fields do not match the v1 schema")
        raw_fitted = payload["fitted_parameters"]
        if not isinstance(raw_fitted, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "value", "lower", "upper"}
            for item in raw_fitted
        ):
            raise ValueError("Twin candidate fitted parameters must be JSON objects")
        try:
            fit = cls(
                schema=str(payload["schema"]),
                incumbent_snapshot_identity=str(payload["incumbent_snapshot_identity"]),
                parameter_estimate_identity=str(payload["parameter_estimate_identity"]),
                probe_bank_identity=str(payload["probe_bank_identity"]),
                parameter_family=str(payload["parameter_family"]),
                fitted_parameters=tuple(
                    (
                        str(item["name"]),
                        float(item["value"]),
                        float(item["lower"]),
                        float(item["upper"]),
                    )
                    for item in raw_fitted
                ),
                retained_parameters=tuple(
                    str(item) for item in payload["retained_parameters"]
                ),
                unidentified_parameters=tuple(
                    str(item) for item in payload["unidentified_parameters"]
                ),
            )
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("Invalid Twin candidate fit") from error
        if fit.to_dict() != dict(payload):
            raise ValueError("Twin candidate fit is not in canonical v1 form")
        return fit


@dataclass(frozen=True)
class TwinFittedCandidate(QPUDigitalTwin):
    """A QPU Twin candidate whose fitted origin is immutable and inspectable."""

    fit: TwinCandidateFit

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.fit, TwinCandidateFit):
            raise TypeError("fit must be a TwinCandidateFit")
        if self.snapshot.identity == self.fit.incumbent_snapshot_identity:
            raise ValueError("candidate snapshot must differ from the incumbent")
        profile = self.noise_model.device_profile
        if profile is None or not profile.source.endswith(
            _SOURCE_MARKER + self.fit.identity
        ):
            raise ValueError("candidate noise model does not bind its fit provenance")


def _parameter_values(
    estimate: TwinParameterEstimate,
) -> dict[str, tuple[float, float, float]]:
    intervals = {
        name: (lower, upper) for name, lower, upper in estimate.confidence_intervals
    }
    return {
        name: (value, *intervals[name]) for name, value in estimate.parameter_values
    }


def _retained_parameter_names(
    profile: DeviceNoiseProfile,
    physical_qubits: tuple[int, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for logical, calibration in enumerate(profile.qubits):
        physical = physical_qubits[logical]
        names.extend(
            (
                f"t1[Q{physical}]",
                f"t2[Q{physical}]",
                f"excited_population[Q{physical}]",
            )
        )
    for duration in profile.gate_durations:
        if duration.wires is None:
            scope = "all"
        else:
            scope = "->".join(f"Q{physical_qubits[index]}" for index in duration.wires)
        names.append(f"gate_duration[{duration.gate_name};{scope}]")
    return tuple(sorted(names))


def _gate_names(
    profile: DeviceNoiseProfile,
    *,
    arity: int,
    logical_qubits: tuple[int, ...],
) -> tuple[str, ...]:
    names: set[str] = set()
    for duration in profile.gate_durations:
        schema = get_operator_schema(duration.gate_name)
        if schema is None or schema.arity != arity:
            continue
        if duration.wires is None or duration.wires == logical_qubits:
            names.add(duration.gate_name)
    return tuple(sorted(names))


def _candidate_profile(
    profile: DeviceNoiseProfile,
    values: Mapping[str, tuple[float, float, float]],
    physical_qubits: tuple[int, ...],
    *,
    captured_at: str,
    fit_identity: str,
) -> DeviceNoiseProfile:
    calibrations: list[QubitNoiseCalibration] = []
    for logical, calibration in enumerate(profile.qubits):
        physical = physical_qubits[logical]
        readout = values[f"readout_bit_flip[Q{physical}]"][0]
        calibrations.append(
            replace(
                calibration,
                readout_error=ReadoutError(
                    ((1.0 - readout, readout), (readout, 1.0 - readout))
                ),
            )
        )
    return replace(
        profile,
        qubits=tuple(calibrations),
        captured_at=captured_at,
        source=profile.source + _SOURCE_MARKER + fit_identity,
    )


def fit_candidate(
    incumbent: QPUDigitalTwin,
    estimate: TwinParameterEstimate,
    *,
    captured_at: str,
) -> TwinFittedCandidate:
    """Create a later immutable candidate without hardware or incumbent mutation.

    The restricted ``local_pauli_readout`` burden model becomes symmetric
    readout flips plus local depolarizing channels. One-qubit and two-qubit
    burden values are converted to depolarizing probabilities by ``3/2`` and
    ``5/4`` respectively so their computational-basis TV burdens agree.

    Examples:
        candidate = fq.twin.fit_candidate(
            incumbent,
            estimate,
            captured_at="2026-08-15T10:30:00+08:00",
        )
        assert candidate.snapshot.identity != incumbent.snapshot.identity
        print(candidate.fit.parameter_estimate_identity)
    """

    if not isinstance(incumbent, QPUDigitalTwin):
        raise TypeError("incumbent must be a QPUDigitalTwin")
    if not isinstance(estimate, TwinParameterEstimate):
        raise TypeError("estimate must be a TwinParameterEstimate")
    profile = incumbent.noise_model.device_profile
    if not isinstance(profile, DeviceNoiseProfile):
        raise ValueError("incumbent requires a frozen device noise profile")
    target = f"{incumbent.snapshot.provider}:{incumbent.snapshot.backend_name}"
    if estimate.target != target:
        raise ValueError("parameter estimate target does not match the incumbent")
    if estimate.physical_qubits != incumbent.snapshot.physical_qubits:
        raise ValueError("parameter estimate mapping does not match the incumbent")
    if estimate.snapshot_identity != incumbent.snapshot.identity:
        raise ValueError("parameter estimate does not belong to the incumbent snapshot")
    try:
        candidate_time = datetime.fromisoformat(captured_at)
    except (TypeError, ValueError) as error:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from error
    if candidate_time.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    if candidate_time <= datetime.fromisoformat(incumbent.snapshot.captured_at):
        raise ValueError("candidate captured_at must be later than the incumbent")

    values = _parameter_values(estimate)
    expected_names = set()
    for physical in incumbent.snapshot.physical_qubits:
        expected_names.update(
            {
                f"readout_bit_flip[Q{physical}]",
                f"single_qubit_pauli[Q{physical}]",
            }
        )
    for name in values:
        match = _TWO_PATTERN.fullmatch(name)
        if match is not None:
            source = int(match.group(1))
            target_wire = int(match.group(2))
            if (
                source == target_wire
                or source not in incumbent.snapshot.physical_qubits
                or target_wire not in incumbent.snapshot.physical_qubits
            ):
                raise ValueError(
                    "two-qubit fitted parameters must use the incumbent mapping"
                )
            expected_names.add(name)
    if set(values) != expected_names:
        raise ValueError("estimate parameters do not cover one supported local family")

    retained = _retained_parameter_names(profile, incumbent.snapshot.physical_qubits)
    fitted = tuple(
        (name, value, lower, upper) for name, (value, lower, upper) in values.items()
    )
    fit = TwinCandidateFit(
        incumbent_snapshot_identity=incumbent.snapshot.identity,
        parameter_estimate_identity=estimate.identity,
        probe_bank_identity=estimate.probe_bank_identity,
        parameter_family=estimate.parameter_family,
        fitted_parameters=fitted,
        retained_parameters=retained,
        unidentified_parameters=estimate.unidentified_parameters,
    )
    candidate_profile = _candidate_profile(
        profile,
        values,
        incumbent.snapshot.physical_qubits,
        captured_at=candidate_time.isoformat(),
        fit_identity=fit.identity,
    )
    model = NoiseModel.from_device_profile(candidate_profile)
    physical_to_logical = {
        physical: logical
        for logical, physical in enumerate(incumbent.snapshot.physical_qubits)
    }
    for name, (burden, _, _) in values.items():
        single = _SINGLE_PATTERN.fullmatch(name)
        if single is not None:
            physical = int(single.group(1))
            logical = physical_to_logical[physical]
            probability = 1.5 * burden
            if probability > 1.0 + 1e-12:
                raise ValueError(
                    f"{name} is outside the representable depolarizing range"
                )
            gates = _gate_names(candidate_profile, arity=1, logical_qubits=(logical,))
            if burden and not gates:
                raise ValueError(f"candidate profile has no one-qubit gates for {name}")
            if burden:
                model.add(
                    gates,
                    depolarizing_channel(min(1.0, probability)),
                    qubits=(logical,),
                )
            continue
        two = _TWO_PATTERN.fullmatch(name)
        if two is not None:
            source = physical_to_logical[int(two.group(1))]
            target_wire = physical_to_logical[int(two.group(2))]
            probability = 1.25 * burden
            if probability > 1.0 + 1e-12:
                raise ValueError(
                    f"{name} is outside the representable depolarizing range"
                )
            gates = _gate_names(
                candidate_profile,
                arity=2,
                logical_qubits=(source, target_wire),
            )
            if burden and not gates:
                raise ValueError(f"candidate profile has no two-qubit gates for {name}")
            if burden:
                model.add(
                    gates,
                    two_qubit_depolarizing_channel(min(1.0, probability)),
                    qubits=(source, target_wire),
                )
            continue
        if _READOUT_PATTERN.fullmatch(name) is None:
            raise ValueError(f"unsupported fitted parameter {name!r}")

    snapshot = TwinSnapshot(
        provider=incumbent.snapshot.provider,
        backend_name=incumbent.snapshot.backend_name,
        captured_at=candidate_profile.captured_at,
        physical_qubits=incumbent.snapshot.physical_qubits,
        calibration_identity=candidate_profile.identity,
        noise_model_identity=model.identity,
    )
    return TwinFittedCandidate(snapshot=snapshot, noise_model=model, fit=fit)


__all__ = ("fit_candidate", "TwinCandidateFit", "TwinFittedCandidate")
