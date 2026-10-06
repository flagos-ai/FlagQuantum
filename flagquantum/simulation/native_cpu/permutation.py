"""Dispatch boundary for native CPU statevector permutations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import torch

from .adjoint import _load_extension, native_cpu_compact_cx_index_available

_CPU_CX_ADJOINT_CYCLE_MINIMUM_LENGTH = 8
_CPU_CX_ADJOINT_CYCLE_BYTES_PER_AMPLITUDE = 9


def compact_cx_permutation_images(
    controls: Sequence[int],
    targets: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Build one compact basis image per qubit for an inverse CX mapping."""

    if len(controls) != len(targets):
        raise ValueError("a CX sequence needs one target per control")
    images = [1 << (n_qubits - qubit - 1) for qubit in range(n_qubits)]
    for control, target in zip(reversed(controls), reversed(targets), strict=True):
        control_mask = 1 << (n_qubits - int(control) - 1)
        target_mask = 1 << (n_qubits - int(target) - 1)
        images = [
            image ^ target_mask if image & control_mask else image for image in images
        ]
    return torch.tensor(images, dtype=torch.int64)


def compact_cx_rzz_swap_images(
    controls: Sequence[int],
    targets: Sequence[int],
    swap_qubits: tuple[int, int],
    n_qubits: int,
) -> torch.Tensor:
    """Build compact inverse images for a CX sequence followed by one SWAP."""

    if len(controls) != len(targets):
        raise ValueError("a CX sequence needs one target per control")
    left, right = (int(qubit) for qubit in swap_qubits)
    left_mask = 1 << (n_qubits - left - 1)
    right_mask = 1 << (n_qubits - right - 1)
    images = [1 << (n_qubits - qubit - 1) for qubit in range(n_qubits)]
    for index, image in enumerate(images):
        left_set = bool(image & left_mask)
        right_set = bool(image & right_mask)
        if left_set != right_set:
            images[index] = image ^ left_mask ^ right_mask
    for control, target in zip(reversed(controls), reversed(targets), strict=True):
        control_mask = 1 << (n_qubits - int(control) - 1)
        target_mask = 1 << (n_qubits - int(target) - 1)
        images = [
            image ^ target_mask if image & control_mask else image for image in images
        ]
    return torch.tensor(images, dtype=torch.int64)


def use_compact_cpu_cx_mapping(n_qubits: int) -> bool:
    """Use compact metadata once a full int32 CX table reaches 16 MiB."""

    return n_qubits >= 22 and native_cpu_compact_cx_index_available()


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


def native_cpu_cx_rzz_swap_available() -> bool:
    """Return whether fused static CX/RZZ/SWAP execution is available."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_RZZ_SWAP", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def native_cpu_clifford_matching_available() -> bool:
    """Return whether the fused inference-only Clifford kernel is loadable."""

    return _load_extension()


def fused_clifford_matching_out(
    state: torch.Tensor,
    cx_mapping: torch.Tensor,
    cz_edges: Sequence[tuple[int, int]] | None,
    n_qubits: int,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply one disjoint CX/CZ matching in a single native CPU pass."""

    phase_encoded = cz_edges is None
    normalized_cz_edges = tuple(
        (int(left), int(right)) for left, right in (cz_edges or ())
    )
    occupied = tuple(qubit for edge in normalized_cz_edges for qubit in edge)
    if (
        not native_cpu_clifford_matching_available()
        or state.requires_grad
        or state.device.type != "cpu"
        or cx_mapping.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or cx_mapping.dtype not in {torch.int32, torch.int64}
        or state.ndim != 2
        or state.shape[1] != 1 << n_qubits
        or cx_mapping.ndim != 1
        or cx_mapping.numel() not in {n_qubits, state.shape[1]}
        or len(set(occupied)) != len(occupied)
        or any(not 0 <= qubit < n_qubits for qubit in occupied)
        or not state.is_contiguous()
        or not cx_mapping.is_contiguous()
        or (
            output is not None
            and (
                output.device.type != "cpu"
                or output.dtype != state.dtype
                or output.shape != state.shape
                or not output.is_contiguous()
                or output.data_ptr() == state.data_ptr()
            )
        )
    ):
        return None
    edge_tensor = torch.tensor(normalized_cz_edges, dtype=torch.int64).reshape(-1, 2)
    if output is None:
        output = torch.empty_like(state)
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_clifford_matching_out(
            state, cx_mapping, edge_tensor, output, n_qubits, phase_encoded
        )
    return output


def native_cpu_cx_adjoint_inplace_available() -> bool:
    """Return whether the zero-state-scratch adjoint CX path is available."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_ADJOINT_INPLACE", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def native_cpu_cx_adjoint_cycles_available() -> bool:
    """Return whether compact in-place CX cycles are enabled and loadable."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_ADJOINT_CYCLES", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and native_cpu_cx_adjoint_inplace_available()
    )


def use_compact_cpu_cx_adjoint_cycles(cx_count: int, n_qubits: int) -> bool:
    """Use compact cycles for a sufficiently long representable CX segment."""

    return (
        cx_count >= _CPU_CX_ADJOINT_CYCLE_MINIMUM_LENGTH
        and 1 < n_qubits < 31
        and native_cpu_cx_adjoint_cycles_available()
    )


def compact_cpu_cx_adjoint_auxiliary_bytes(n_qubits: int) -> int:
    """Conservative cycle-index allocation bound for one amplitude row."""

    if not 1 < n_qubits < 31:
        raise ValueError("compact CPU CX cycles require 2 to 30 qubits")
    return _CPU_CX_ADJOINT_CYCLE_BYTES_PER_AMPLITUDE * (1 << n_qubits)


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


def fused_compact_cx_rzz_swap_out(
    state: torch.Tensor,
    images: torch.Tensor,
    output: torch.Tensor,
    *,
    rzz_qubits: tuple[int, int],
    rzz_angle: float,
) -> bool:
    """Apply one static CX/RZZ/SWAP segment into a reusable output."""

    if (
        not native_cpu_cx_rzz_swap_available()
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
    first, second = (int(qubit) for qubit in rzz_qubits)
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_compact_cx_rzz_swap_out(
            state,
            images,
            output,
            first,
            second,
            float(rzz_angle),
        )
    return True


def fused_cx_rzz_swap_sequence_inplace_(
    state: torch.Tensor,
    controls: Sequence[int],
    targets: Sequence[int],
    n_qubits: int,
    *,
    rzz_qubits: tuple[int, int],
    rzz_angle: float,
    swap_qubits: tuple[int, int],
) -> bool:
    """Apply one static CX/RZZ/SWAP segment in place without state scratch."""

    if (
        not native_cpu_cx_rzz_swap_available()
        or state.requires_grad
        or state.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or state.ndim != 2
        or not 1 < n_qubits < 63
        or state.shape[1] != 1 << n_qubits
        or len(controls) != len(targets)
        or not state.is_contiguous()
    ):
        return False
    first, second = (int(qubit) for qubit in rzz_qubits)
    swap_left, swap_right = (int(qubit) for qubit in swap_qubits)
    control_tensor = torch.tensor(tuple(controls), dtype=torch.int64)
    target_tensor = torch.tensor(tuple(targets), dtype=torch.int64)
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_cx_rzz_swap_sequence_inplace_(
            state,
            control_tensor,
            target_tensor,
            n_qubits,
            first,
            second,
            swap_left,
            swap_right,
            float(rzz_angle),
        )
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
    n_qubits: int,
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
        or ket.shape[1] != 1 << n_qubits
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
            n_qubits,
        )
    return True


def fused_compact_cx_adjoint_inplace_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    images: torch.Tensor,
) -> bool:
    """Apply a compact inverse CX mapping in place through permutation cycles."""

    if (
        not native_cpu_cx_adjoint_cycles_available()
        or ket.requires_grad
        or adjoint.requires_grad
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or images.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or images.dtype != torch.int64
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or images.ndim != 1
        or not 1 < images.numel() < 31
        or ket.shape[1] != 1 << images.numel()
        or not all(item.is_contiguous() for item in (ket, adjoint, images))
    ):
        return False
    with torch.no_grad():
        torch.ops.flagquantum_native.fused_compact_cx_adjoint_inplace_(
            ket,
            adjoint,
            images,
        )
    return True
