"""Fail-closed access to FlagTree-owned kernel extensions."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any

import torch

from .provenance import triton_compiler_provenance

_TLE_BACKEND_ALIASES = {"cuda": "nvidia"}


@lru_cache(maxsize=None)
def require_flagtree_tle_primitive(primitive: str, *, backend: str) -> str:
    """Require one TLE primitive and return its FlagTree registry backend.

    FlagTree's runtime target calls the NVIDIA backend ``cuda``, while its TLE
    whitelist is registered under ``nvidia``. Resolving that alias here keeps
    provider-specific naming out of the semantic catalog.
    """

    distribution, version, integration_path, status = triton_compiler_provenance()
    if (
        distribution != "flagtree"
        or integration_path != "flagtree"
        or status != "resolved"
    ):
        identity = distribution or "missing"
        raise RuntimeError(
            "FlagTree TLE requires the flagtree distribution to own the "
            f"triton namespace; found {identity!r}"
        )

    registry_backend = _TLE_BACKEND_ALIASES.get(backend, backend)
    try:
        tle: Any = import_module("triton.experimental.tle")
        tle.require_tle(registry_backend, primitive)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"FlagTree {version or 'unknown'} does not provide TLE primitive "
            f"{primitive!r} for backend {registry_backend!r}"
        ) from exc
    return registry_backend


def apply_complex64_local_1q_tle(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    bit_position: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a local one-qubit matrix with FlagTree's TLE async vector loads."""

    amplitude_count = state.shape[1] if state.ndim == 2 else 0
    if (
        state.device.type != "cuda"
        or state.dtype != torch.complex64
        or state.ndim != 2
        or not state.is_contiguous()
        or amplitude_count == 0
        or amplitude_count & (amplitude_count - 1)
    ):
        raise ValueError(
            "FlagTree TLE local 1q requires contiguous CUDA complex64 [B, 2**n]"
        )
    if matrix.shape != (2, 2):
        raise ValueError("FlagTree TLE local 1q requires a 2x2 matrix")
    if not 0 <= bit_position < amplitude_count.bit_length() - 1:
        raise ValueError("bit_position is outside the local state address")

    matrix = matrix.to(device=state.device, dtype=state.dtype).contiguous()
    output = torch.empty_like(state) if output is None else output
    if (
        output.shape != state.shape
        or output.dtype != state.dtype
        or output.device != state.device
        or not output.is_contiguous()
    ):
        raise ValueError("FlagTree TLE local 1q output must match the input state")

    require_flagtree_tle_primitive("load", backend=state.device.type)
    kernel: Any = import_module(
        ".triton.extensions.tle.statevector_gates",
        package=__package__,
    )
    return kernel.launch_complex64_local_1q_tle(
        state,
        matrix,
        bit_position=bit_position,
        output=output,
    )


__all__ = ("apply_complex64_local_1q_tle", "require_flagtree_tle_primitive")
