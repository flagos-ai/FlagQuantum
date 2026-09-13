"""Concise construction entry points for QPU digital twins."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..noise import NoiseModel
from .model import QPUDigitalTwin


def _split_target(target: str) -> tuple[str, str]:
    if not isinstance(target, str):
        raise TypeError("target must be a string in the form 'provider:backend'")
    provider, separator, backend = target.partition(":")
    provider = provider.strip()
    backend = backend.strip()
    if separator != ":" or not provider or not backend:
        raise ValueError("target must use the form 'provider:backend'")
    return provider.lower(), backend


def from_noise_model(
    noise_model: NoiseModel,
    *,
    target: str,
    qubits: Sequence[int],
) -> QPUDigitalTwin:
    """Build a mapped QPU digital twin from a device-backed noise model."""

    provider, backend = _split_target(target)
    return QPUDigitalTwin.from_noise_model(
        noise_model,
        provider=provider,
        backend_name=backend,
        physical_qubits=qubits,
    )


def from_quafu_chip_info(
    chip_info: Mapping[str, Any],
    *,
    target: str,
    qubits: Sequence[int],
    readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
) -> QPUDigitalTwin:
    """Build a mapped QPU digital twin from native Quafu calibration data."""

    provider, backend = _split_target(target)
    if provider != "quafu":
        raise ValueError("from_quafu_chip_info requires target='quafu:<backend>'")
    return QPUDigitalTwin.from_quafu_chip_info(
        chip_info,
        backend_name=backend,
        physical_qubits=qubits,
        readout_confusion_matrices=readout_confusion_matrices,
        correlated_readout_confusion_matrix=correlated_readout_confusion_matrix,
    )


__all__ = ("from_noise_model", "from_quafu_chip_info")
