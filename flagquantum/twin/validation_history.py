"""Longitudinal validation history for frozen QPU Twins."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import QPUDigitalTwin
from .series import TwinValidationSeries

_HISTORY_SCHEMA = "flagquantum.twin_validation_history.v1"
_HISTORY_FIELDS = {
    "schema",
    "provider",
    "backend_name",
    "physical_qubits",
    "circuit_identity",
    "snapshot_identities",
    "captured_at",
    "observation_count",
    "mean_twin_qpu_agreements",
    "mean_ideal_qpu_agreements",
    "mean_qpu_repeatabilities",
    "simultaneous_finite_shot_tv_radii",
    "confidence_levels",
    "verified_tv_error_bounds",
    "validation_series",
}


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True)
class TwinValidationHistory:
    """Validation series for one fixed circuit across chronological Twins."""

    provider: str
    backend_name: str
    physical_qubits: tuple[int, ...]
    circuit_identity: str
    snapshot_identities: tuple[str, ...]
    captured_at: tuple[str, ...]
    validation_series: tuple[TwinValidationSeries, ...]
    schema: str = _HISTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HISTORY_SCHEMA:
            raise ValueError("unsupported Twin validation-history schema")
        physical_qubits = tuple(int(qubit) for qubit in self.physical_qubits)
        identities = tuple(self.snapshot_identities)
        timestamps = tuple(self.captured_at)
        series = tuple(self.validation_series)
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("validation history requires a provider and backend")
        if (
            not physical_qubits
            or len(physical_qubits) != len(set(physical_qubits))
            or any(qubit < 0 for qubit in physical_qubits)
        ):
            raise ValueError("physical_qubits must be unique and non-negative")
        if len(self.circuit_identity) != 64 or any(
            character not in "0123456789abcdef" for character in self.circuit_identity
        ):
            raise ValueError("circuit_identity must be a lowercase SHA-256 digest")
        if len(identities) < 2:
            raise ValueError("validation history requires at least two observations")
        if len(set(identities)) != len(identities):
            raise ValueError("Twin snapshot identities must be unique")
        if len(timestamps) != len(identities) or len(series) != len(identities):
            raise ValueError("validation observations must have aligned fields")
        if any(not isinstance(item, TwinValidationSeries) for item in series):
            raise TypeError("validation history requires TwinValidationSeries records")
        parsed = tuple(datetime.fromisoformat(value) for value in timestamps)
        if any(right <= left for left, right in zip(parsed, parsed[1:])):
            raise ValueError(
                "Twin snapshots must have strictly increasing captured_at values"
            )
        expected_scope = (self.circuit_identity, physical_qubits)
        for index, item in enumerate(series):
            if item.snapshot_identity != identities[index]:
                raise ValueError("validation series does not match its Twin snapshot")
            if (item.circuit_identity, item.physical_qubits) != expected_scope:
                raise ValueError("validation series must use one circuit and mapping")
        structures = {
            (item.supported_operations, item.maximum_instruction_count)
            for item in series
        }
        if len(structures) != 1:
            raise ValueError("validation series circuit structures differ")
        reports = tuple(
            report_identity
            for item in series
            for report_identity in item.report_identities
        )
        if len(reports) != len(set(reports)):
            raise ValueError("validation history requires distinct hardware reports")
        object.__setattr__(self, "physical_qubits", physical_qubits)
        object.__setattr__(self, "snapshot_identities", identities)
        object.__setattr__(self, "captured_at", timestamps)
        object.__setattr__(self, "validation_series", series)

    @property
    def observation_count(self) -> int:
        return len(self.validation_series)

    @property
    def mean_twin_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.mean_twin_qpu_agreement for item in self.validation_series)

    @property
    def mean_ideal_qpu_agreements(self) -> tuple[float, ...]:
        return tuple(item.mean_ideal_qpu_agreement for item in self.validation_series)

    @property
    def mean_qpu_repeatabilities(self) -> tuple[float | None, ...]:
        return tuple(item.mean_qpu_repeatability for item in self.validation_series)

    @property
    def simultaneous_finite_shot_tv_radii(self) -> tuple[float, ...]:
        return tuple(
            item.simultaneous_finite_shot_tv_radius for item in self.validation_series
        )

    @property
    def confidence_levels(self) -> tuple[float, ...]:
        return tuple(item.confidence_level for item in self.validation_series)

    @property
    def verified_tv_error_bounds(self) -> tuple[float, ...]:
        return tuple(item.verified_tv_error_bound for item in self.validation_series)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "physical_qubits": list(self.physical_qubits),
            "circuit_identity": self.circuit_identity,
            "snapshot_identities": list(self.snapshot_identities),
            "captured_at": list(self.captured_at),
            "observation_count": self.observation_count,
            "mean_twin_qpu_agreements": list(self.mean_twin_qpu_agreements),
            "mean_ideal_qpu_agreements": list(self.mean_ideal_qpu_agreements),
            "mean_qpu_repeatabilities": list(self.mean_qpu_repeatabilities),
            "simultaneous_finite_shot_tv_radii": list(
                self.simultaneous_finite_shot_tv_radii
            ),
            "confidence_levels": list(self.confidence_levels),
            "verified_tv_error_bounds": list(self.verified_tv_error_bounds),
            "validation_series": [item.to_dict() for item in self.validation_series],
        }


def build_validation_history(
    observations: Sequence[tuple[QPUDigitalTwin, TwinValidationSeries]],
) -> TwinValidationHistory:
    """Bind one fixed circuit's validation series across frozen Twins.

    Examples:
        history = fq.twin.build_validation_history(
            [(reference_twin, reference_series), (current_twin, current_series)]
        )
        print(history.mean_twin_qpu_agreements)

    Raises:
        TypeError: If observations are not Twin and validation-series pairs.
        ValueError: If fewer than two comparable observations are supplied.
    """

    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must pair Twins with validation series")
    pairs = tuple(observations)
    if len(pairs) < 2:
        raise ValueError("validation history requires at least two observations")
    if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
        raise TypeError("observations must contain (Twin, validation series) pairs")
    twins = tuple(pair[0] for pair in pairs)
    series = tuple(pair[1] for pair in pairs)
    if any(not isinstance(twin, QPUDigitalTwin) for twin in twins) or any(
        not isinstance(item, TwinValidationSeries) for item in series
    ):
        raise TypeError(
            "observations must pair Twins with TwinValidationSeries records"
        )

    first = twins[0].snapshot
    target = (first.provider, first.backend_name, first.physical_qubits)
    for twin, item in zip(twins, series, strict=True):
        snapshot = twin.snapshot
        if (
            snapshot.provider,
            snapshot.backend_name,
            snapshot.physical_qubits,
        ) != target:
            raise ValueError("validation history requires one target and mapping")
        if item.snapshot_identity != snapshot.identity:
            raise ValueError("validation series does not match its Twin snapshot")

    return TwinValidationHistory(
        provider=first.provider,
        backend_name=first.backend_name,
        physical_qubits=first.physical_qubits,
        circuit_identity=series[0].circuit_identity,
        snapshot_identities=tuple(twin.snapshot.identity for twin in twins),
        captured_at=tuple(twin.snapshot.captured_at for twin in twins),
        validation_series=series,
    )


def load_validation_history(
    path: str | PathLike[str],
) -> TwinValidationHistory:
    """Load one validation history offline without contacting a provider.

    Examples:
        history = fq.twin.load_validation_history("validation-history.json")

    Raises:
        ValueError: If the file is not a canonical v1 validation history.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Twin validation history from {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Twin validation-history file must contain a JSON object")
    actual = set(payload)
    if actual != _HISTORY_FIELDS:
        raise ValueError(
            "Twin validation-history fields do not match the v1 schema: "
            f"missing={sorted(_HISTORY_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _HISTORY_FIELDS)}"
        )
    series_payloads = payload["validation_series"]
    if not isinstance(series_payloads, list) or any(
        not isinstance(item, Mapping) for item in series_payloads
    ):
        raise ValueError("Twin validation-history series must be JSON objects")
    try:
        history = TwinValidationHistory(
            schema=str(payload["schema"]),
            provider=str(payload["provider"]),
            backend_name=str(payload["backend_name"]),
            physical_qubits=tuple(payload["physical_qubits"]),
            circuit_identity=str(payload["circuit_identity"]),
            snapshot_identities=tuple(payload["snapshot_identities"]),
            captured_at=tuple(payload["captured_at"]),
            validation_series=tuple(
                TwinValidationSeries.from_dict(item) for item in series_payloads
            ),
        )
        if _canonical(history.to_dict()) != _canonical(payload):
            raise ValueError("Twin validation history is not in canonical v1 form")
        return history
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Invalid Twin validation history") from error


def dump_validation_history(
    history: TwinValidationHistory,
    path: str | PathLike[str],
) -> None:
    """Write one private history file without replacing different content.

    Examples:
        fq.twin.dump_validation_history(history, "validation-history.json")

    Raises:
        TypeError: If ``history`` is not a Twin validation history.
        ValueError: If the destination cannot be written safely.
    """

    if not isinstance(history, TwinValidationHistory):
        raise TypeError("history must be a TwinValidationHistory")
    destination = Path(path)
    encoded = _canonical(history.to_dict()) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        try:
            existing = load_validation_history(destination)
        except ValueError as error:
            raise ValueError(
                "Refusing to replace invalid Twin validation history at "
                f"{destination}"
            ) from error
        if _canonical(existing.to_dict()) != _canonical(history.to_dict()):
            raise ValueError(
                "Refusing to replace different Twin validation history at "
                f"{destination}"
            )
    except OSError as error:
        raise ValueError(
            f"Cannot write Twin validation history to {destination}"
        ) from error


__all__ = (
    "build_validation_history",
    "dump_validation_history",
    "load_validation_history",
    "TwinValidationHistory",
)
