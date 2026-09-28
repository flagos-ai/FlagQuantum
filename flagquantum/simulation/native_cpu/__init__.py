"""Optional package-local CPU operators with explicit PyTorch fallbacks."""

from .adjoint import (
    fused_rotation_adjoint_,
    fused_rzz_segment_adjoint_,
    native_cpu_adjoint_available,
)

__all__ = [
    "fused_rotation_adjoint_",
    "fused_rzz_segment_adjoint_",
    "native_cpu_adjoint_available",
]
