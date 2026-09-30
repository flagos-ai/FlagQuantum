"""Dispatch boundary for native CPU statevector permutations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import torch

from .adjoint import _load_extension, native_cpu_compact_cx_index_available


def compact_cx_permutation_images(
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Build one compact basis image per wire for an inverse CX mapping."""

    if len(controls) != len(targets):
        raise ValueError("a CX sequence needs one target per control")
    images = [1 << (n_wires - wire - 1) for wire in range(n_wires)]
    for control, target in zip(reversed(controls), reversed(targets), strict=True):
        control_mask = 1 << (n_wires - int(control) - 1)
        target_mask = 1 << (n_wires - int(target) - 1)
        images = [
            image ^ target_mask if image & control_mask else image for image in images
        ]
    return torch.tensor(images, dtype=torch.int64)


def use_compact_cpu_cx_mapping(n_wires: int) -> bool:
    """Use compact metadata once a full int32 CX table reaches 16 MiB."""

    return n_wires >= 22 and native_cpu_compact_cx_index_available()


def native_cpu_cx_adjoint_gather_available() -> bool:
    """Return whether the dual-state CX gather is enabled and loadable."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_ADJOINT_GATHER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def native_cpu_cx_gather_available() -> bool:
    """Return whether the single-state CX gather is enabled and loadable."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_GATHER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def native_cpu_cx_adjoint_inplace_available() -> bool:
    """Return whether the zero-state-scratch adjoint CX path is available."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_ADJOINT_INPLACE", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def fused_cx_gather_out(
    state: torch.Tensor,
    index: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Gather a detached CPU state into a reusable output buffer."""

    if (
        not native_cpu_cx_gather_available()
        or state.requires_grad
        or state.device.type != "cpu"
        or index.device.type != "cpu"
        or output.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or output.dtype != state.dtype
        or index.dtype not in {torch.int32, torch.int64}
        or state.ndim != 2
        or output.shape != state.shape
        or index.shape != (state.shape[1],)
        or not all(item.is_contiguous() for item in (state, index, output))
        or state.data_ptr() == output.data_ptr()
    ):
        return False
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_cx_gather_out(state, index, output)
    return True


def fused_compact_cx_gather_out(
    state: torch.Tensor,
    images: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Gather a CPU state from compact linear CX metadata."""

    if (
        os.getenv("FQ_NATIVE_CPU_COMPACT_CX_INDEX", "0").strip().lower()
        in {"0", "false", "off", "no"}
        or not _load_extension()
        or state.requires_grad
        or state.device.type != "cpu"
        or images.device.type != "cpu"
        or output.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or output.dtype != state.dtype
        or images.dtype != torch.int64
        or state.ndim != 2
        or output.shape != state.shape
        or images.ndim != 1
        or state.shape[1] != 1 << images.numel()
        or not all(item.is_contiguous() for item in (state, images, output))
        or state.data_ptr() == output.data_ptr()
    ):
        return False
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_compact_cx_gather_out(state, images, output)
    return True


def fused_cx_adjoint_gather(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather ket and adjoint with one shared permutation-table traversal."""

    if (
        not native_cpu_cx_adjoint_gather_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or index.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or index.dtype not in {torch.int32, torch.int64}
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or index.shape != (ket.shape[1],)
        or not all(item.is_contiguous() for item in (ket, adjoint, index))
    ):
        return None
    with torch.no_grad():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            torch.ops.flagquantum_native.fused_cx_adjoint_gather(ket, adjoint, index),
        )


def fused_compact_cx_adjoint_gather(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather ket and adjoint from a compact linear CX mapping."""

    if (
        os.getenv("FQ_NATIVE_CPU_COMPACT_CX_INDEX", "0").strip().lower()
        in {"0", "false", "off", "no"}
        or not _load_extension()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or images.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or images.dtype != torch.int64
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or images.ndim != 1
        or not all(item.is_contiguous() for item in (ket, adjoint, images))
    ):
        return None
    with torch.no_grad():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            torch.ops.flagquantum_native.fused_compact_cx_adjoint_gather(
                ket, adjoint, images
            ),
        )


def fused_cx_adjoint_inplace_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_wires: int,
) -> bool:
    """Apply an inverse CX sequence to ket and adjoint without state scratch."""

    if (
        not native_cpu_cx_adjoint_inplace_available()
        or ket.requires_grad
        or adjoint.requires_grad
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or ket.shape[1] != 1 << n_wires
        or len(controls) != len(targets)
        or not ket.is_contiguous()
        or not adjoint.is_contiguous()
    ):
        return False
    control_tensor = torch.tensor(tuple(controls), dtype=torch.int64)
    target_tensor = torch.tensor(tuple(targets), dtype=torch.int64)
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_cx_adjoint_inplace_(
            ket,
            adjoint,
            control_tensor,
            target_tensor,
            n_wires,
        )
    return True
