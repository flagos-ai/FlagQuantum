"""Catalog authorization and rollout policy for bounded local SWAP sequences."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-014-A"
_EVIDENCED_STATE_SHAPES = frozenset(
    {
        (1, 1 << 16),
        (1, 1 << 20),
        (1, 1 << 24),
        (4, 1 << 20),
    }
)


def _swap_sequence_dispatch_enabled() -> bool:
    """Return whether the measured SV-014 public route is enabled."""

    return os.getenv("FQ_TRITON_SWAP_SEQUENCE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _swap_sequence_shape_supported(
    state_shape: Sequence[int],
    *,
    swaps: Sequence[tuple[int, int]],
    n_qubits: int,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    swaps = tuple((int(first), int(second)) for first, second in swaps)
    n_qubits = int(n_qubits)
    if len(state_shape) != 2 or n_qubits < 2:
        return False
    _, amplitudes = state_shape
    return bool(
        state_shape in _EVIDENCED_STATE_SHAPES
        and amplitudes == 1 << n_qubits
        and 4 <= len(swaps) <= 8
        and all(
            first != second and 0 <= first < n_qubits and 0 <= second < n_qubits
            for first, second in swaps
        )
    )


def _swap_sequence_kernel_enabled(
    state: torch.Tensor,
    *,
    swaps: Sequence[tuple[int, int]],
    n_qubits: int,
) -> bool:
    """Return whether SV-014 supports this exact public runtime request."""

    return bool(
        _swap_sequence_dispatch_enabled()
        and _swap_sequence_shape_supported(
            state.shape,
            swaps=swaps,
            n_qubits=n_qubits,
        )
        and state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
        and not state.requires_grad
    )


def _swap_sequence_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the SWAP-sequence kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.swap_sequence.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_swap_sequence_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-014 implementation or fail closed."""

    return _require_cataloged_kernel(
        _swap_sequence_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="bounded local SWAP-sequence kernel",
    )


def _apply_cataloged_swap_sequence(
    state: torch.Tensor,
    *,
    swaps: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Execute cataloged SV-014 after exact authorization."""

    _require_swap_sequence_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_swap import (
        apply_complex64_local_swap_sequence,
    )

    return apply_complex64_local_swap_sequence(state, swaps=swaps)


def _try_apply_cataloged_swap_sequence(
    state: torch.Tensor,
    *,
    swaps: tuple[tuple[int, int], ...],
    n_qubits: int,
) -> torch.Tensor | None:
    """Route an evidenced SWAP sequence, or preserve the reference path."""

    if not _swap_sequence_kernel_enabled(
        state,
        swaps=swaps,
        n_qubits=n_qubits,
    ):
        return None
    return _apply_cataloged_swap_sequence(state, swaps=swaps)


__all__ = (
    "_apply_cataloged_swap_sequence",
    "_require_swap_sequence_kernel",
    "_swap_sequence_dispatch_enabled",
    "_swap_sequence_kernel_enabled",
    "_swap_sequence_kernel_match",
    "_swap_sequence_shape_supported",
    "_try_apply_cataloged_swap_sequence",
)
