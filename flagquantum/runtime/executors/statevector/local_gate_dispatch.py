"""Catalog-backed dispatch policy for the local single- and two-qubit kernels.

The plain local gates are the narrowest Triton family the statevector executor
connects, and two independent rules decide whether one is reached. The first is
a measured shape window per kernel, which a deployment can override by name.
The second is the flat local addressing contract: a shard past it must stay on
the index-based eager path, and that rule is not a window over a shape but a
bound over an element count, so a shard inside a window is still refused when it
is past the contract.

Both rules are declarations about a kernel rather than work the executor does,
so they live here beside the other families' policy modules and the forward
sweep, the reversible adjoint sweep, and their tests read them from one place.
"""

from __future__ import annotations

import os

import torch

from ....kernels.catalog import KernelRequest
from ....kernels.triton import FLAT_LOCAL_MAX_AMPLITUDES
from .kernel_dispatch import KernelDecision, select_cataloged_triton_kernel

_TRITON_LOCAL_1Q_DEFAULT_SHAPES = frozenset((1, 1 << e) for e in (10, 16, 20, 24))
_TRITON_LOCAL_CX_DEFAULT_SHAPES = frozenset({(1, 1 << 24)})


def _flat_local_address_supported(amplitudes: torch.Tensor) -> bool:
    """Whether a Triton flat local-gate kernel may address this shard.

    The local kernels derive one linear element offset per amplitude pair, so a
    state past their address contract stays on the index-based eager path
    instead of faulting the CUDA context.
    """

    return amplitudes.numel() <= FLAT_LOCAL_MAX_AMPLITUDES


def _triton_local_1q_requested(shape: tuple[int, int]) -> bool:
    """Select the measured default window, while retaining an explicit override."""

    configured = os.getenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q")
    if configured is None:
        return shape in _TRITON_LOCAL_1Q_DEFAULT_SHAPES
    return configured.strip().lower() in {"1", "true", "on", "yes"}


def _triton_local_cx_requested(shape: tuple[int, int]) -> bool:
    configured = os.getenv("FQ_STATEVECTOR_TRITON_LOCAL_CX")
    if configured is None:
        return shape in _TRITON_LOCAL_CX_DEFAULT_SHAPES
    return configured.strip().lower() in {"1", "true", "on", "yes"}


def _triton_local_1q_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
    addressable: bool = True,
    shape: tuple[int, int],
) -> KernelDecision:
    return select_cataloged_triton_kernel(
        "local_1q",
        request=KernelRequest(
            semantic_id="statevector.apply.matrix_1q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-001-A",
        requested=_triton_local_1q_requested(shape),
        runtime_supported=runtime_supported and addressable,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_local_cx_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
    addressable: bool = True,
    shape: tuple[int, int],
) -> KernelDecision:
    return select_cataloged_triton_kernel(
        "local_cx",
        request=KernelRequest(
            semantic_id="statevector.apply.cnot.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-002-A",
        requested=_triton_local_cx_requested(shape),
        runtime_supported=runtime_supported and addressable,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_local_cx_enabled(
    *, device_type: str, dtype: str, addressable: bool = True, shape: tuple[int, int]
) -> bool:
    return bool(
        _triton_local_cx_decision(
            device_type=device_type,
            dtype=dtype,
            addressable=addressable,
            shape=shape,
        ).accelerated
    )


__all__ = (
    "_flat_local_address_supported",
    "_triton_local_1q_decision",
    "_triton_local_1q_requested",
    "_triton_local_cx_decision",
    "_triton_local_cx_enabled",
    "_triton_local_cx_requested",
)
