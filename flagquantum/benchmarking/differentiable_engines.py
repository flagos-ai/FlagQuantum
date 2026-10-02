"""Engine metadata for differentiable simulator benchmarks."""

from __future__ import annotations

from typing import Literal

EngineName = Literal[
    "flagquantum_native",
    "pennylane_default_qubit",
    "flagquantum_adjoint",
    "flagquantum_adjoint_python_fallback",
    "flagquantum_adjoint_gather_rollback",
    "flagquantum_adjoint_forward_cx_rollback",
    "flagquantum_adjoint_observable_cache_rollback",
    "flagquantum_adjoint_rotation_tile_rollback",
    "flagquantum_adjoint_euler_triple_rollback",
    "flagquantum_adjoint_observable_boundary_rollback",
    "flagquantum_adjoint_observable_rotation_rollback",
    "flagquantum_adjoint_shared_rzz_rollback",
    "flagquantum_adjoint_forward_rzz_rotation_rollback",
    "flagquantum_adjoint_forward_wide_tile_rollback",
    "flagquantum_adjoint_flat_pair_simd_rollback",
    "flagquantum_adjoint_cx_rotation_fusion_rollback",
    "flagquantum_adjoint_terminal_no_restore_rollback",
    "flagquantum_adjoint_compact_cx_index_rollback",
    "flagquantum_adjoint_saved_parameter_revalidation",
    "pennylane_lightning_adjoint",
]

ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_native",
    "pennylane_default_qubit",
)
ADJOINT_ENGINE_NAMES: tuple[EngineName, ...] = (
    "flagquantum_adjoint",
    "flagquantum_adjoint_python_fallback",
    "pennylane_lightning_adjoint",
)
ALL_ENGINE_NAMES = (
    ENGINE_NAMES
    + ADJOINT_ENGINE_NAMES
    + (
        "flagquantum_adjoint_gather_rollback",
        "flagquantum_adjoint_forward_cx_rollback",
        "flagquantum_adjoint_observable_cache_rollback",
        "flagquantum_adjoint_rotation_tile_rollback",
        "flagquantum_adjoint_euler_triple_rollback",
        "flagquantum_adjoint_observable_boundary_rollback",
        "flagquantum_adjoint_observable_rotation_rollback",
        "flagquantum_adjoint_shared_rzz_rollback",
        "flagquantum_adjoint_forward_rzz_rotation_rollback",
        "flagquantum_adjoint_forward_wide_tile_rollback",
        "flagquantum_adjoint_flat_pair_simd_rollback",
        "flagquantum_adjoint_cx_rotation_fusion_rollback",
        "flagquantum_adjoint_terminal_no_restore_rollback",
        "flagquantum_adjoint_compact_cx_index_rollback",
        "flagquantum_adjoint_saved_parameter_revalidation",
    )
)
