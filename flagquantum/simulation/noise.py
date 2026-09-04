"""Compatibility facade for native noise and density-matrix execution.

New code should import noise semantics from :mod:`flagquantum.noise`, lowering
from :mod:`flagquantum.compilation.noise`, and numerical execution from
:mod:`flagquantum.simulation.density_matrix`.
"""

from ..compilation.noise import channel_instruction, lower_noise_model
from ..noise import (
    CorrelatedReadoutError,
    DeviceNoiseProfile,
    GateDuration,
    KrausChannel,
    NoiseModel,
    NoiseRule,
    QubitNoiseCalibration,
    ReadoutError,
    ReadoutRule,
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
from ..runtime.noise_registry import noisy_density_matrix
from .density_matrix import (
    apply_kraus_density,
    apply_unitary_density,
    density_matrix,
    density_matrix_from_ir,
    expand_operator,
    expectation_z_density,
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
    "apply_kraus_density",
    "apply_unitary_density",
    "bit_flip_channel",
    "coherent_overrotation_channel",
    "channel_instruction",
    "density_matrix",
    "density_matrix_from_ir",
    "depolarizing_channel",
    "expectation_z_density",
    "expand_operator",
    "lower_noise_model",
    "noisy_density_matrix",
    "phase_flip_channel",
    "phase_damping_channel",
    "reset_error_channel",
    "thermal_relaxation_channel",
    "two_qubit_depolarizing_channel",
)
