"""Optional package-local CPU operators with explicit PyTorch fallbacks."""

from .adjoint import (
    fused_rotation_adjoint_,
    fused_rotation_segment_adjoint_,
    fused_rzz_segment_adjoint_,
    native_cpu_adjoint_available,
    native_cpu_rotation_segment_available,
)
from .rotation import fused_rotation_block_forward_, native_cpu_rotation_available

__all__ = [
    "fused_rotation_adjoint_",
    "fused_rotation_segment_adjoint_",
    "fused_rzz_segment_adjoint_",
    "native_cpu_adjoint_available",
    "native_cpu_rotation_segment_available",
    "fused_rotation_block_forward_",
    "native_cpu_rotation_available",
]
