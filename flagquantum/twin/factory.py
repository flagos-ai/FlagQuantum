"""Concise construction entry points for QPU digital twins."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..noise import NoiseModel
from .model import QPUDigitalTwin


def from_noise_model(
    noise_model: NoiseModel,
    *,
    provider: str,
    backend: str,
    qubits: Sequence[int],
) -> QPUDigitalTwin:
    """Build a mapped QPU digital twin from a device-backed noise model."""

    return QPUDigitalTwin.from_noise_model(
        noise_model,
        provider=provider,
        backend_name=backend,
        physical_qubits=qubits,
    )


def from_quafu_chip_info(
    chip_info: Mapping[str, Any],
    *,
    backend: str,
    qubits: Sequence[int],
    readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
) -> QPUDigitalTwin:
    """Build a mapped QPU digital twin from native Quafu calibration data."""

    return QPUDigitalTwin.from_quafu_chip_info(
        chip_info,
        backend_name=backend,
        physical_qubits=qubits,
        readout_confusion_matrices=readout_confusion_matrices,
        correlated_readout_confusion_matrix=correlated_readout_confusion_matrix,
    )


__all__ = ("from_noise_model", "from_quafu_chip_info")
