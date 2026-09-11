"""Frozen QPU digital-twin models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch

from ..core.ir import ensure_circuit_ir
from ..noise import NoiseModel, noisy_density_matrix
from .prediction import TwinPrediction
from .validation import _total_variation

_SNAPSHOT_SCHEMA = "flagquantum.twin_snapshot.v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TwinSnapshot:
    """Immutable identity of one mapped QPU calibration model."""

    provider: str
    backend_name: str
    captured_at: str
    physical_qubits: tuple[int, ...]
    calibration_identity: str
    noise_model_identity: str
    schema: str = _SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SNAPSHOT_SCHEMA:
            raise ValueError("unsupported twin snapshot schema")
        if not self.provider.strip() or not self.backend_name.strip():
            raise ValueError("twin snapshot requires a provider and backend name")
        captured = datetime.fromisoformat(self.captured_at)
        if captured.tzinfo is None:
            raise ValueError("twin snapshot captured_at must include a timezone")
        if (
            not self.physical_qubits
            or len(self.physical_qubits) != len(set(self.physical_qubits))
            or any(wire < 0 for wire in self.physical_qubits)
        ):
            raise ValueError(
                "physical_qubits must be non-empty, unique, and non-negative"
            )
        for name, value in (
            ("calibration_identity", self.calibration_identity),
            ("noise_model_identity", self.noise_model_identity),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "backend_name": self.backend_name,
            "captured_at": self.captured_at,
            "physical_qubits": list(self.physical_qubits),
            "calibration_identity": self.calibration_identity,
            "noise_model_identity": self.noise_model_identity,
        }


@dataclass(frozen=True)
class QPUDigitalTwin:
    """A frozen, calibration-conditioned model of one physical QPU mapping."""

    snapshot: TwinSnapshot
    noise_model: NoiseModel

    def __post_init__(self) -> None:
        if self.noise_model.identity != self.snapshot.noise_model_identity:
            raise ValueError("noise model does not match the frozen twin snapshot")
        profile = self.noise_model.device_profile
        if profile is None or profile.identity != self.snapshot.calibration_identity:
            raise ValueError(
                "device calibration does not match the frozen twin snapshot"
            )

    @classmethod
    def from_noise_model(
        cls,
        noise_model: NoiseModel,
        *,
        provider: str,
        backend_name: str,
        physical_qubits: Sequence[int],
    ) -> "QPUDigitalTwin":
        """Freeze a device-backed noise model as a mapped QPU digital twin."""

        if not isinstance(noise_model, NoiseModel):
            raise TypeError("noise_model must be a NoiseModel")
        frozen_model = NoiseModel.from_dict(noise_model.to_dict())
        profile = frozen_model.device_profile
        if profile is None:
            raise ValueError("a QPU digital twin requires a device noise profile")
        selected = tuple(int(wire) for wire in physical_qubits)
        logical_wires = {calibration.wire for calibration in profile.qubits}
        if logical_wires != set(range(len(selected))):
            raise ValueError(
                "device profile wires must match the logical order of physical_qubits"
            )
        snapshot = TwinSnapshot(
            provider=str(provider).strip(),
            backend_name=str(backend_name).strip(),
            captured_at=profile.captured_at,
            physical_qubits=selected,
            calibration_identity=profile.identity,
            noise_model_identity=frozen_model.identity,
        )
        return cls(snapshot=snapshot, noise_model=frozen_model)

    @classmethod
    def from_quafu_chip_info(
        cls,
        chip_info: Mapping[str, Any],
        *,
        backend_name: str,
        physical_qubits: Sequence[int],
        readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
        correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
    ) -> "QPUDigitalTwin":
        """Build and freeze a twin from one Quafu calibration response."""

        from ..remote.qpu import quafu_noise_model_from_chip_info

        selected = tuple(int(wire) for wire in physical_qubits)
        model = quafu_noise_model_from_chip_info(
            chip_info,
            physical_qubits=selected,
            readout_confusion_matrices=readout_confusion_matrices,
            correlated_readout_confusion_matrix=correlated_readout_confusion_matrix,
        )
        return cls.from_noise_model(
            model,
            provider="quafu",
            backend_name=backend_name,
            physical_qubits=selected,
        )

    @classmethod
    def from_quafu_provider(
        cls,
        provider: Any,
        *,
        backend_name: str,
        physical_qubits: Sequence[int],
        readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
        correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
    ) -> "QPUDigitalTwin":
        """Fetch, convert, and freeze one Quafu calibration snapshot."""

        if getattr(provider, "provider", None) != "quafu" or not hasattr(
            provider, "fetch_chip_info"
        ):
            raise TypeError("from_quafu_provider requires a Quafu provider")
        name = str(backend_name).strip()
        if not name:
            raise ValueError("backend_name cannot be empty")
        return cls.from_quafu_chip_info(
            provider.fetch_chip_info(name),
            backend_name=name,
            physical_qubits=physical_qubits,
            readout_confusion_matrices=readout_confusion_matrices,
            correlated_readout_confusion_matrix=correlated_readout_confusion_matrix,
        )

    def predict(self, circuit: Any) -> TwinPrediction:
        """Predict ideal and calibration-conditioned output probabilities."""

        if self.noise_model.identity != self.snapshot.noise_model_identity:
            raise RuntimeError("the frozen twin noise model was modified")
        if not hasattr(circuit, "probabilities") or not hasattr(circuit, "to_ir"):
            raise TypeError("QPUDigitalTwin.predict requires an fq.Circuit")
        ir = ensure_circuit_ir(circuit)
        if ir.n_wires != len(self.snapshot.physical_qubits):
            raise ValueError("circuit width must match the twin physical mapping")

        ideal_tensor = (
            torch.as_tensor(circuit.probabilities()).detach().cpu().reshape(-1)
        )
        if ideal_tensor.numel() != 2**ir.n_wires:
            raise ValueError("QPUDigitalTwin.predict supports one circuit batch")
        density = noisy_density_matrix(circuit, self.noise_model)
        predicted_tensor = (
            torch.diagonal(density, dim1=-2, dim2=-1).real.detach().cpu().reshape(-1)
        )
        predicted_tensor = self.noise_model.apply_readout_probabilities(
            predicted_tensor, n_wires=ir.n_wires
        )
        ideal = tuple(float(value) for value in ideal_tensor)
        predicted = tuple(float(value) for value in predicted_tensor)
        return TwinPrediction(
            snapshot_identity=self.snapshot.identity,
            circuit_identity=ir.content_hash,
            n_wires=ir.n_wires,
            ideal_probabilities=ideal,
            twin_probabilities=predicted,
            total_variation_from_ideal=_total_variation(ideal, predicted),
        )


__all__ = ("QPUDigitalTwin", "TwinSnapshot")
