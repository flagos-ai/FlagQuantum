"""Catalog authorization for fused MPS one-site contractions."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _opt_in_environment_flag, _require_cataloged_kernel
from ..real_imag_kernels import complex_einsum_pair

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-003-A"


def _mps_one_site_rollout_enabled() -> bool:
    """Return whether the default MPS-003 route has not been disabled."""

    return _opt_in_environment_flag("FQ_TRITON_MPS_ONE_SITE", "1")


def _mps_two_site_rollout_enabled() -> bool:
    """Return whether the opt-in MPS two-site fused route is enabled."""

    return _opt_in_environment_flag("FQ_TRITON_MPS_TWO_SITE")


def _mps_one_site_kernel_enabled(tensor: torch.Tensor, gate: torch.Tensor) -> bool:
    """Return whether the default route supports this exact tensor pair."""

    contraction_volume = int(tensor.shape[0] * tensor.shape[1] * tensor.shape[3])
    return bool(
        _mps_one_site_rollout_enabled()
        and tensor.is_cuda
        and tensor.dtype == torch.complex64
        and gate.dtype == torch.complex64
        and gate.device == tensor.device
        and contraction_volume >= 2**12
    )


def _mps_one_site_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the fused one-site contraction against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.contract.one_site_gate",
            device=device_type,
            dtype=dtype,
            layout="mps_one_site",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_mps_one_site_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _mps_one_site_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused MPS one-site kernel",
    )


def _apply_cataloged_mps_one_site(
    tensor: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Execute the fused contraction after exact catalog authorization."""

    _require_mps_one_site_kernel(
        device_type=tensor.device.type,
        dtype=str(tensor.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_one_site import fused_mps_one_site

    return fused_mps_one_site(tensor, gate)


def _apply_mps_one_site(
    tensor: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """Select the cataloged path only inside its evidenced default window."""

    if _mps_one_site_kernel_enabled(tensor, gate):
        return _apply_cataloged_mps_one_site(tensor, gate), True
    equation = "pq,blqr->blpr" if gate.ndim == 2 else "bpq,blqr->blpr"
    return complex_einsum_pair(equation, gate, tensor), False


def _try_apply_cataloged_mps_one_site_bucket(
    tensors: torch.Tensor, gates: torch.Tensor
) -> torch.Tensor | None:
    """Flatten a spatial bucket and route it when MPS-003 authorizes the shape."""

    if tensors.ndim != 5 or int(tensors.shape[3]) != 2:
        raise ValueError(
            "MPS one-site bucket must have shape [sites,batch,left,2,right]"
        )
    sites, batch, left_dim, _, right_dim = tensors.shape
    if gates.shape != (sites, batch, 2, 2):
        raise ValueError("MPS one-site bucket gates must have shape [sites,batch,2,2]")
    flat_tensors = tensors.reshape(sites * batch, left_dim, 2, right_dim)
    flat_gates = gates.reshape(sites * batch, 2, 2)
    if not _mps_one_site_kernel_enabled(flat_tensors, flat_gates):
        return None
    output = _apply_cataloged_mps_one_site(flat_tensors, flat_gates)
    return output.reshape(sites, batch, left_dim, 2, right_dim)


__all__ = (
    "_apply_cataloged_mps_one_site",
    "_apply_mps_one_site",
    "_mps_one_site_kernel_match",
    "_mps_one_site_kernel_enabled",
    "_mps_one_site_rollout_enabled",
    "_mps_two_site_rollout_enabled",
    "_require_mps_one_site_kernel",
    "_try_apply_cataloged_mps_one_site_bucket",
)
