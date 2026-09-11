"""Backend-neutral noise channels and model semantics."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .channels import (
    KrausChannel,
    amplitude_damping_channel,
    bit_flip_channel,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
)
from .device_profile import DeviceNoiseProfile, GateDuration, QubitNoiseCalibration
from .model import (
    CorrelatedReadoutError,
    NoiseModel,
    NoiseRule,
    ReadoutError,
    ReadoutRule,
)

__all__ = (
    "CorrelatedReadoutError",
    "KrausChannel",
    "DeviceNoiseProfile",
    "GateDuration",
    "NoiseModel",
    "NoiseRule",
    "ReadoutError",
    "ReadoutRule",
    "QubitNoiseCalibration",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "coherent_overrotation_channel",
    "depolarizing_channel",
    "phase_damping_channel",
    "phase_flip_channel",
    "reset_error_channel",
    "thermal_relaxation_channel",
    "two_qubit_depolarizing_channel",
    "noisy_density_matrix",
)


def __getattr__(name: str) -> Any:
    if name == "noisy_density_matrix":
        return getattr(
            import_module("flagquantum.runtime.noise_registry"),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
