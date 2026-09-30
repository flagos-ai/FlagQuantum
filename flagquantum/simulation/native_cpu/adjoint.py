"""Dispatch boundary for the native single-qubit CPU adjoint operator."""

from __future__ import annotations

import os
from importlib import import_module
from typing import cast

import torch

_GATE_KINDS = {"rx": 0, "ry": 1, "rz": 2, "rzz": 3}
_EXTENSION_LOADED = False
_EXTENSION_ERROR: Exception | None = None


def _enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_ADJOINT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _rotation_segment_enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_ROTATION_SEGMENT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _wide_rotation_tiles_enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _rotation_segment_tile_wires() -> int:
    """Return the native rotation tile width, including the legacy rollback."""

    return 11 if _wide_rotation_tiles_enabled() else 2


def _rotation_pair_fast_path_enabled() -> bool:
    return os.getenv(
        "FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _flat_rotation_pair_simd_enabled() -> bool:
    return os.getenv(
        "FQ_NATIVE_CPU_ADJOINT_FLAT_PAIR_SIMD", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def native_cpu_cx_rotation_adjoint_fusion_available() -> bool:
    """Return whether a CX permutation may feed a native rotation adjoint."""

    return os.getenv(
        "FQ_NATIVE_CPU_ADJOINT_CX_ROTATION_FUSION", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def native_cpu_compact_cx_index_available() -> bool:
    """Return whether CPU CX mappings may use compact basis images."""

    return (
        os.getenv("FQ_NATIVE_CPU_COMPACT_CX_INDEX", "0").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def native_cpu_terminal_adjoint_no_restore_available() -> bool:
    """Return whether the earliest Euler layer may omit unused state restoration."""

    return (
        os.getenv("FQ_NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _rotation_pair_fast_path_enabled()
        and _flat_rotation_pair_simd_enabled()
    )


def native_cpu_shared_rotation_gradient_available() -> bool:
    """Return whether shared rotation segments may aggregate one VJP in native code."""

    return os.getenv(
        "FQ_NATIVE_CPU_SHARED_ROTATION_GRADIENT", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def native_cpu_rotation_rzz_fusion_available() -> bool:
    """Return whether a terminal rotation layer may absorb a shared RZZ layer."""

    return os.getenv(
        "FQ_NATIVE_CPU_ADJOINT_RX_RZZ_FUSION", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def native_cpu_adjoint_rzz_h_fusion_available() -> bool:
    """Return whether a fused RZZ boundary may absorb preceding Hadamards."""

    return os.getenv("FQ_NATIVE_CPU_ADJOINT_RZZ_H_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def native_cpu_observable_rotation_boundary_available() -> bool:
    """Return whether the observable seed may be fused into reverse rotations."""

    return os.getenv(
        "FQ_NATIVE_CPU_OBSERVABLE_ROTATION_BOUNDARY", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def native_cpu_rotation_tile_wires() -> int:
    """Return the active native adjoint rotation tile width."""

    return _rotation_segment_tile_wires()


def _load_extension() -> bool:
    global _EXTENSION_ERROR, _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return True
    if _EXTENSION_ERROR is not None:
        return False
    try:
        import_module("flagquantum.simulation.native_cpu._C")
    except (ImportError, OSError) as error:
        _EXTENSION_ERROR = error
        return False
    _EXTENSION_LOADED = True
    return True


def native_cpu_adjoint_available() -> bool:
    """Return whether the installed native operator is enabled and loadable."""

    return _enabled() and _load_extension()


def fused_observable_adjoint_seed(
    ket: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor | None:
    """Build ``2 * ket * weights`` in one native CPU traversal."""

    real_dtype = torch.float32 if ket.dtype == torch.complex64 else torch.float64
    if (
        os.getenv("FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY", "1").strip().lower()
        in {"0", "false", "off", "no"}
        or not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or weights.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or weights.dtype != real_dtype
        or ket.ndim != 2
        or weights.shape != (ket.shape[1],)
        or not ket.is_contiguous()
        or not weights.is_contiguous()
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_observable_adjoint_seed(ket, weights),
        )


def fused_observable_expectation(
    ket: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor | None:
    """Evaluate a real diagonal observable in one native CPU traversal."""

    real_dtype = torch.float32 if ket.dtype == torch.complex64 else torch.float64
    if (
        os.getenv("FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY", "1").strip().lower()
        in {"0", "false", "off", "no"}
        or not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or weights.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or weights.dtype != real_dtype
        or ket.ndim != 2
        or weights.shape != (ket.shape[1],)
        or not ket.is_contiguous()
        or not weights.is_contiguous()
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_observable_expectation(ket, weights),
        )


def native_cpu_parallel_build_available() -> bool:
    """Return whether native operators were built with Torch intra-op parallelism."""

    if not _load_extension():
        return False
    extension = import_module("flagquantum.simulation.native_cpu._C")
    probe = getattr(extension, "parallel_build_available", None)
    return bool(probe is not None and probe())


def native_cpu_rotation_segment_available() -> bool:
    """Return whether the optional multi-gate CPU adjoint path is available."""

    return _rotation_segment_enabled() and native_cpu_adjoint_available()


def fused_rotation_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    *,
    name: str,
    wire: int,
    second_wire: int | None = None,
    n_wires: int,
) -> torch.Tensor | None:
    """Mutate ket and adjoint to the pre-gate state and return one VJP.

    ``None`` is the stable fallback signal. The caller retains the existing
    PyTorch implementation for disabled, unavailable, or unsupported cases.
    """

    gate_kind = _GATE_KINDS.get(name)
    if (
        gate_kind is None
        or not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or matrix.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or matrix.dtype != ket.dtype
        or (name == "rzz" and second_wire is None)
        or not ket.is_contiguous()
        or not adjoint.is_contiguous()
        or not matrix.is_contiguous()
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rotation_adjoint_(
                ket,
                adjoint,
                matrix,
                wire,
                -1 if second_wire is None else second_wire,
                n_wires,
                gate_kind,
            ),
        )


def _fused_rotation_segment_adjoint_result(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    angles: torch.Tensor,
    gate_kinds: torch.Tensor,
    wires: torch.Tensor,
    *,
    n_wires: int,
    aggregate_shared_parameter: bool = False,
    rzz_angles: torch.Tensor | None = None,
    rzz_first_wires: torch.Tensor | None = None,
    rzz_second_wires: torch.Tensor | None = None,
    fuse_preceding_hadamards: bool = False,
    observable_weights: torch.Tensor | None = None,
    cx_index: torch.Tensor | None = None,
    cx_images: torch.Tensor | None = None,
    restore_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run the native rotation adjoint with an optional preceding CX gather."""

    rzz_tensors = (rzz_angles, rzz_first_wires, rzz_second_wires)
    if (
        not native_cpu_rotation_segment_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or angles.device.type != "cpu"
        or gate_kinds.device.type != "cpu"
        or wires.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or angles.dtype
        != (torch.float32 if ket.dtype == torch.complex64 else torch.float64)
        or gate_kinds.dtype != torch.int64
        or wires.dtype != torch.int64
        or angles.ndim != 1
        or angles.shape[0] < 2
        or angles.shape[0] > (256 if _wide_rotation_tiles_enabled() else 48)
        or gate_kinds.shape != angles.shape
        or wires.shape != angles.shape
        or not all(
            item.is_contiguous() for item in (ket, adjoint, angles, gate_kinds, wires)
        )
        or (
            any(item is None for item in rzz_tensors)
            and any(item is not None for item in rzz_tensors)
        )
        or (
            rzz_angles is not None
            and (
                not native_cpu_rotation_rzz_fusion_available()
                or rzz_angles.dtype != angles.dtype
                or rzz_first_wires is None
                or rzz_second_wires is None
                or rzz_first_wires.dtype != torch.int64
                or rzz_second_wires.dtype != torch.int64
                or rzz_angles.ndim != 1
                or rzz_first_wires.shape != rzz_angles.shape
                or rzz_second_wires.shape != rzz_angles.shape
                or not rzz_angles.is_contiguous()
                or not rzz_first_wires.is_contiguous()
                or not rzz_second_wires.is_contiguous()
            )
        )
        or (
            observable_weights is not None
            and (
                not native_cpu_observable_rotation_boundary_available()
                or observable_weights.device.type != "cpu"
                or observable_weights.dtype != angles.dtype
                or observable_weights.shape != (ket.shape[1],)
                or not observable_weights.is_contiguous()
            )
        )
        or (
            cx_index is not None
            and (
                not native_cpu_cx_rotation_adjoint_fusion_available()
                or cx_index.device.type != "cpu"
                or cx_index.dtype not in {torch.int32, torch.int64}
                or cx_index.shape != (ket.shape[1],)
                or not cx_index.is_contiguous()
            )
        )
        or (
            cx_images is not None
            and (
                not native_cpu_compact_cx_index_available()
                or cx_images.device.type != "cpu"
                or cx_images.dtype != torch.int64
                or cx_images.shape != (n_wires,)
                or not cx_images.is_contiguous()
            )
        )
        or (cx_index is not None and cx_images is not None)
    ):
        return None
    with torch.no_grad():
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.ops.flagquantum_native.fused_rotation_segment_adjoint_(
                ket,
                adjoint,
                angles,
                gate_kinds,
                wires,
                n_wires,
                aggregate_shared_parameter,
                _rotation_segment_tile_wires(),
                _rotation_pair_fast_path_enabled(),
                _flat_rotation_pair_simd_enabled(),
                restore_state,
                fuse_preceding_hadamards
                and native_cpu_adjoint_rzz_h_fusion_available(),
                rzz_angles,
                rzz_first_wires,
                rzz_second_wires,
                observable_weights,
                cx_index,
                cx_images,
            ),
        )


def fused_rotation_segment_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    angles: torch.Tensor,
    gate_kinds: torch.Tensor,
    wires: torch.Tensor,
    *,
    n_wires: int,
    aggregate_shared_parameter: bool = False,
    rzz_angles: torch.Tensor | None = None,
    rzz_first_wires: torch.Tensor | None = None,
    rzz_second_wires: torch.Tensor | None = None,
    fuse_preceding_hadamards: bool = False,
    observable_weights: torch.Tensor | None = None,
    restore_state: bool = True,
) -> torch.Tensor | None:
    """Undo a multi-wire RX/RY/RZ segment and return one VJP per gate."""

    result = _fused_rotation_segment_adjoint_result(
        ket,
        adjoint,
        angles,
        gate_kinds,
        wires,
        n_wires=n_wires,
        aggregate_shared_parameter=aggregate_shared_parameter,
        rzz_angles=rzz_angles,
        rzz_first_wires=rzz_first_wires,
        rzz_second_wires=rzz_second_wires,
        fuse_preceding_hadamards=fuse_preceding_hadamards,
        observable_weights=observable_weights,
        restore_state=restore_state,
    )
    return None if result is None else result[0]


def fused_cx_rotation_segment_adjoint(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    cx_index: torch.Tensor | None,
    angles: torch.Tensor,
    gate_kinds: torch.Tensor,
    wires: torch.Tensor,
    *,
    n_wires: int,
    aggregate_shared_parameter: bool = False,
    observable_weights: torch.Tensor | None = None,
    cx_images: torch.Tensor | None = None,
    restore_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Undo a CX segment and following rotation segment in one native boundary."""

    return _fused_rotation_segment_adjoint_result(
        ket,
        adjoint,
        angles,
        gate_kinds,
        wires,
        n_wires=n_wires,
        aggregate_shared_parameter=aggregate_shared_parameter,
        observable_weights=observable_weights,
        cx_index=cx_index,
        cx_images=cx_images,
        restore_state=restore_state,
    )


def fused_rzz_segment_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    angles: torch.Tensor,
    first_wires: torch.Tensor,
    second_wires: torch.Tensor,
    *,
    n_wires: int,
    aggregate_shared_parameter: bool = False,
) -> torch.Tensor | None:
    """Undo a commuting RZZ segment and return one VJP per gate."""

    real_dtype = torch.float32 if ket.dtype == torch.complex64 else torch.float64
    if (
        not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or angles.device.type != "cpu"
        or angles.dtype != real_dtype
        or first_wires.device.type != "cpu"
        or second_wires.device.type != "cpu"
        or first_wires.dtype != torch.int64
        or second_wires.dtype != torch.int64
        or angles.numel() < 2
        or not all(
            item.is_contiguous()
            for item in (ket, adjoint, angles, first_wires, second_wires)
        )
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rzz_segment_adjoint_(
                ket,
                adjoint,
                angles,
                first_wires,
                second_wires,
                n_wires,
                aggregate_shared_parameter,
            ),
        )
