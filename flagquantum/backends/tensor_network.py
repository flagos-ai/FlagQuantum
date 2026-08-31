"""Stable expert interfaces for tensor-network-specific outputs."""

from ..simulation.tensor_execution import (
    tensor_network_amplitude,
    tensor_network_amplitudes,
    tensor_network_expectations,
)
from . import run_tensor_network

__all__ = (
    "run_tensor_network",
    "tensor_network_amplitude",
    "tensor_network_amplitudes",
    "tensor_network_expectations",
)
