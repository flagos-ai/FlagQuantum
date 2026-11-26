"""Backend-neutral noise channels and model semantics."""

from __future__ import annotations

import sys as _alias_sys
from importlib import import_module
from typing import Any

from flagquantum.core.noise import (
    CorrelatedReadoutError,
    NoiseModel,
    NoiseRule,
    ReadoutError,
    ReadoutRule,
)
from flagquantum.core.noise_profile import (
    DeviceNoiseProfile,
    GateDuration,
    QubitNoiseCalibration,
)
from flagquantum.core.operators.channels import (
    CHANNEL_FACTORIES,
    KrausChannel,
    UnitaryMixture,
    amplitude_damping_channel,
    bit_flip_channel,
    channel_from_parameters,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
    y_flip_channel,
)

__all__ = (
    "CHANNEL_FACTORIES",
    "CorrelatedReadoutError",
    "KrausChannel",
    "DeviceNoiseProfile",
    "GateDuration",
    "NoiseModel",
    "NoiseRule",
    "QubitNoiseCalibration",
    "ReadoutError",
    "ReadoutRule",
    "UnitaryMixture",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "channel_from_parameters",
    "coherent_overrotation_channel",
    "depolarizing_channel",
    "phase_damping_channel",
    "phase_flip_channel",
    "reset_error_channel",
    "thermal_relaxation_channel",
    "two_qubit_depolarizing_channel",
    "y_flip_channel",
    "noisy_density_matrix",
)


def __getattr__(name: str) -> Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    if name == "noisy_density_matrix":
        return getattr(
            import_module("flagquantum.runtime.noise_registry"),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
