"""Concise construction entry points for QPU digital twins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..noise import NoiseModel
from .model import QPUDigitalTwin, TwinSnapshot


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
    """Build a mapped QPU digital twin from a device-backed noise model.

    Examples:
        twin = fq.twin.from_noise_model(
            device_noise_model,
            target="acme:research-qpu",
            qubits=(12, 13),
        )
    """

    provider, backend = _split_target(target)
    if not isinstance(noise_model, NoiseModel):
        raise TypeError("noise_model must be a NoiseModel")
    frozen_model = NoiseModel.from_dict(noise_model.to_dict())
    profile = frozen_model.device_profile
    if profile is None:
        raise ValueError("a QPU digital twin requires a device noise profile")
    selected = tuple(int(qubit) for qubit in qubits)
    logical_wires = {calibration.wire for calibration in profile.qubits}
    if logical_wires != set(range(len(selected))):
        raise ValueError("device profile wires must match the logical order of qubits")
    snapshot = TwinSnapshot(
        provider=provider,
        backend_name=backend,
        captured_at=profile.captured_at,
        physical_qubits=selected,
        calibration_identity=profile.identity,
        noise_model_identity=frozen_model.identity,
    )
    return QPUDigitalTwin(snapshot=snapshot, noise_model=frozen_model)


def from_quafu_chip_info(
    chip_info: Mapping[str, Any],
    *,
    target: str,
    qubits: Sequence[int],
    readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
) -> QPUDigitalTwin:
    """Build a mapped QPU digital twin from native Quafu calibration data.

    Examples:
        twin = fq.twin.from_quafu_chip_info(
            chip_info,
            target="quafu:Shenglian",
            qubits=(20, 27),
        )
    """

    provider, backend = _split_target(target)
    if provider != "quafu":
        raise ValueError("from_quafu_chip_info requires target='quafu:<backend>'")
    from ..remote.qpu import quafu_noise_model_from_chip_info

    selected = tuple(int(qubit) for qubit in qubits)
    model = quafu_noise_model_from_chip_info(
        chip_info,
        physical_qubits=selected,
        readout_confusion_matrices=readout_confusion_matrices,
        correlated_readout_confusion_matrix=correlated_readout_confusion_matrix,
    )
    return from_noise_model(
        model,
        target=f"quafu:{backend}",
        qubits=selected,
    )


__all__ = ("from_noise_model", "from_quafu_chip_info")
