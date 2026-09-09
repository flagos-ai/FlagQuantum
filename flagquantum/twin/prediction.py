"""Frozen digital-twin predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .validation import TwinValidationReport, _total_variation

_PREDICTION_SCHEMA = "flagquantum.twin_prediction.v1"


def _validate_probabilities(values: tuple[float, ...], n_wires: int) -> None:
    if len(values) != 2**n_wires:
        raise ValueError("probability vector does not match n_wires")
    if any(not math.isfinite(value) or value < -1e-12 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")


@dataclass(frozen=True)
class TwinPrediction:
    """Ideal and calibration-conditioned predictions for one frozen circuit."""

    snapshot_identity: str
    circuit_identity: str
    n_wires: int
    ideal_probabilities: tuple[float, ...]
    twin_probabilities: tuple[float, ...]
    total_variation_from_ideal: float
    schema: str = _PREDICTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PREDICTION_SCHEMA:
            raise ValueError("unsupported twin prediction schema")
        if len(self.snapshot_identity) != 64 or len(self.circuit_identity) != 64:
            raise ValueError("prediction identities must be SHA-256 digests")
        if self.n_wires < 1:
            raise ValueError("n_wires must be positive")
        _validate_probabilities(self.ideal_probabilities, self.n_wires)
        _validate_probabilities(self.twin_probabilities, self.n_wires)
        expected = _total_variation(self.ideal_probabilities, self.twin_probabilities)
        if not math.isclose(self.total_variation_from_ideal, expected, abs_tol=1e-12):
            raise ValueError(
                "total_variation_from_ideal does not match the probabilities"
            )

    def compare_counts(
        self, counts: Mapping[str, int], *, reverse_bits: bool = False
    ) -> TwinValidationReport:
        """Compare this frozen prediction with hardware measurement counts."""

        bins = [0] * (2**self.n_wires)
        for raw_bitstring, raw_count in counts.items():
            bitstring = str(raw_bitstring)
            if reverse_bits:
                bitstring = bitstring[::-1]
            if len(bitstring) != self.n_wires or set(bitstring) - {"0", "1"}:
                raise ValueError("hardware count keys must be fixed-width bitstrings")
            count = int(raw_count)
            if count < 0:
                raise ValueError("hardware counts must be non-negative")
            bins[int(bitstring, 2)] += count
        shots = sum(bins)
        if shots <= 0:
            raise ValueError("hardware counts must contain at least one shot")
        hardware = tuple(count / shots for count in bins)
        ideal_distance = _total_variation(self.ideal_probabilities, hardware)
        twin_distance = _total_variation(self.twin_probabilities, hardware)
        return TwinValidationReport(
            snapshot_identity=self.snapshot_identity,
            circuit_identity=self.circuit_identity,
            hardware_probabilities=hardware,
            ideal_hardware_total_variation=ideal_distance,
            twin_hardware_total_variation=twin_distance,
            total_variation_improvement=ideal_distance - twin_distance,
            shots=shots,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_identity": self.snapshot_identity,
            "circuit_identity": self.circuit_identity,
            "n_wires": self.n_wires,
            "ideal_probabilities": list(self.ideal_probabilities),
            "twin_probabilities": list(self.twin_probabilities),
            "total_variation_from_ideal": self.total_variation_from_ideal,
        }


__all__ = ("TwinPrediction",)
