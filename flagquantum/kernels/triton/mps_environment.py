"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.kernels.triton.mps_environment import (
    _MAX_BOND as _MAX_BOND,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _MAX_CHANNEL_BOND as _MAX_CHANNEL_BOND,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _MAX_CHANNEL_CONTRACTION_WORK as _MAX_CHANNEL_CONTRACTION_WORK,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _MAX_CHANNELS as _MAX_CHANNELS,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _MAX_CONTRACTION_WORK as _MAX_CONTRACTION_WORK,
)
from flagquantum.simulation.kernels.triton.mps_environment import __all__ as __all__
from flagquantum.simulation.kernels.triton.mps_environment import (
    _launch as _launch,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _launch_channels as _launch_channels,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _mps_environment_channels_kernel as _mps_environment_channels_kernel,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _mps_environment_transfer_kernel as _mps_environment_transfer_kernel,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _reference as _reference,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _reference_channels as _reference_channels,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _supported_channel_shape as _supported_channel_shape,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _supported_shape as _supported_shape,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _validate as _validate,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    _validate_channels as _validate_channels,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    fused_mps_environment_channels as fused_mps_environment_channels,
)
from flagquantum.simulation.kernels.triton.mps_environment import (
    fused_mps_environment_transfer as fused_mps_environment_transfer,
)

_implementation = _import_module(
    "flagquantum.simulation.kernels.triton.mps_environment"
)
_sys.modules[__name__] = _implementation
