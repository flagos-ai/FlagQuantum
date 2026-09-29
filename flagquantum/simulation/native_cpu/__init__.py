"""Optional package-local CPU operators with explicit PyTorch fallbacks."""

from .adjoint import (
    fused_rotation_adjoint_,
    fused_rotation_segment_adjoint_,
    fused_rzz_segment_adjoint_,
    native_cpu_adjoint_available,
    native_cpu_parallel_build_available,
    native_cpu_rotation_segment_available,
    native_cpu_shared_rotation_gradient_available,
)
from .permutation import (
    fused_cx_adjoint_gather,
    fused_cx_gather_out,
    native_cpu_cx_adjoint_gather_available,
    native_cpu_cx_gather_available,
)
from .rotation import (
    fused_rotation_block_forward_,
    native_cpu_one_qubit_layer_available,
    native_cpu_rotation_available,
)
from .rzz import fused_rzz_segment_forward_, native_cpu_rzz_available

__all__ = [
    "fused_rotation_adjoint_",
    "fused_rotation_segment_adjoint_",
    "fused_rzz_segment_adjoint_",
    "native_cpu_adjoint_available",
    "native_cpu_parallel_build_available",
    "native_cpu_rotation_segment_available",
    "native_cpu_shared_rotation_gradient_available",
    "fused_rotation_block_forward_",
    "native_cpu_rotation_available",
    "native_cpu_one_qubit_layer_available",
    "fused_cx_adjoint_gather",
    "fused_cx_gather_out",
    "native_cpu_cx_adjoint_gather_available",
    "native_cpu_cx_gather_available",
    "fused_rzz_segment_forward_",
    "native_cpu_rzz_available",
]
