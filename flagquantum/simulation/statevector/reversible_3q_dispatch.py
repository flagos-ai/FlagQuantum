"""Catalog authorization and rollout policy for local CCX and CSWAP."""

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

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-013-A"
_EVIDENCED_STATE_SHAPES = frozenset(
    {
        (1, 1 << 20),
        (1, 1 << 24),
        (4, 1 << 20),
    }
)
_SUPPORTED_OPCODES = frozenset({"ccx", "cswap"})


def _reversible_3q_dispatch_enabled() -> bool:
    """Return whether the measured SV-013 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_REVERSIBLE_3Q",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _reversible_3q_shape_supported(
    state_shape: Sequence[int],
    *,
    qubits: Sequence[int],
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    qubits = tuple(int(qubit) for qubit in qubits)
    n_qubits = int(n_qubits)
    if len(state_shape) != 2 or n_qubits < 3:
        return False
    _, amplitudes = state_shape
    return bool(
        opcode in _SUPPORTED_OPCODES
        and state_shape in _EVIDENCED_STATE_SHAPES
        and amplitudes == 1 << n_qubits
        and len(qubits) == 3
        and len(set(qubits)) == 3
        and all(0 <= qubit < n_qubits for qubit in qubits)
    )


def _reversible_3q_kernel_enabled(
    state: torch.Tensor,
    *,
    qubits: Sequence[int],
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether SV-013 supports this exact public runtime request."""

    return bool(
        _reversible_3q_dispatch_enabled()
        and _reversible_3q_shape_supported(
            state.shape,
            qubits=qubits,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        and state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
        and not state.requires_grad
    )


def _reversible_3q_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the reversible-permutation kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.reversible_permutation_3q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_reversible_3q_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-013 implementation or fail closed."""

    return _require_cataloged_kernel(
        _reversible_3q_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="reversible three-qubit permutation kernel",
    )


def _apply_cataloged_reversible_3q(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int, int],
    opcode: str,
) -> torch.Tensor:
    """Execute cataloged SV-013 after exact authorization."""

    _require_reversible_3q_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_reversible_3q import (
        apply_complex64_local_reversible_3q,
    )

    return apply_complex64_local_reversible_3q(
        state,
        qubits=qubits,
        operation=opcode,  # type: ignore[arg-type]
    )


def _try_apply_cataloged_reversible_3q(
    state: torch.Tensor,
    *,
    qubits: tuple[int, int, int],
    n_qubits: int,
    opcode: str,
) -> torch.Tensor | None:
    """Route an evidenced CCX or CSWAP, or preserve the reference path."""

    if not _reversible_3q_kernel_enabled(
        state,
        qubits=qubits,
        n_qubits=n_qubits,
        opcode=opcode,
    ):
        return None
    return _apply_cataloged_reversible_3q(
        state,
        qubits=qubits,
        opcode=opcode,
    )


__all__ = (
    "_apply_cataloged_reversible_3q",
    "_require_reversible_3q_kernel",
    "_reversible_3q_dispatch_enabled",
    "_reversible_3q_kernel_enabled",
    "_reversible_3q_kernel_match",
    "_reversible_3q_shape_supported",
    "_try_apply_cataloged_reversible_3q",
)
