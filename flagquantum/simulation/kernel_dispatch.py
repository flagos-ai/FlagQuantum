"""Shared fail-closed authorization for cataloged simulation kernels."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..kernels.catalog import (
    KERNEL_DEVICES,
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)


def _require_cataloged_kernel(
    match: KernelMatchResult,
    *,
    implementation_id: str,
    description: str,
) -> KernelImplementation:
    """Return one exact implementation or report why it is unauthorized."""

    candidate = next(
        (
            item.implementation
            for item in match.candidates
            if item.implementation.implementation_id == implementation_id
        ),
        None,
    )
    if candidate is not None:
        return candidate

    rejection = next(
        (
            item
            for item in match.rejections
            if item.implementation.implementation_id == implementation_id
        ),
        None,
    )
    mismatch_codes = (
        tuple(mismatch.code for mismatch in rejection.mismatches)
        if rejection is not None
        else ("implementation_not_registered",)
    )
    raise RuntimeError(
        f"{description} is not authorized by the kernel catalog: "
        + ", ".join(mismatch_codes)
    )


def _operand_kernel_axis(operands: Sequence[torch.Tensor]) -> tuple[str, str] | None:
    """Return the shared ``(device_type, dtype)`` of a tensor group.

    A group that spans two devices or two precisions has no single catalog
    request to make, and a device outside the declared axis cannot reach a
    cataloged implementation at all. Both cases return ``None`` instead of
    guessing, so a caller's route question has one spelling everywhere. An empty
    group is one more case of the same rule rather than a separate guard: it has
    no device for the group to agree on either.
    """

    if len({operand.device for operand in operands}) != 1:
        return None
    device_type = operands[0].device.type
    if device_type not in KERNEL_DEVICES:
        return None
    dtypes = {str(operand.dtype).removeprefix("torch.") for operand in operands}
    if len(dtypes) != 1:
        return None
    return device_type, dtypes.pop()


def _catalog_declares_kernel(
    semantic_id: str,
    *,
    device_type: str,
    dtype: str,
    layout: str,
) -> bool:
    """Return whether an evidenced implementation covers this device and layout.

    The catalog is the authority for which execution devices can run a fused
    route, so a caller asks this question instead of matching on a device
    literal. The provider is deliberately not pinned: the question is whether
    *any* evidenced implementation covers the device and precision, so a record
    added for another provider changes the answer without editing the caller.
    Local addressing and the forward direction are the shape every dense
    contraction route in this package declares.
    """

    if device_type not in KERNEL_DEVICES:
        return False
    return match_kernel_implementations(
        KernelRequest(
            semantic_id=semantic_id,
            device=device_type,
            dtype=dtype,
            layout=layout,
            direction="forward",
            addressing=("local",),
        )
    ).matched


__all__ = (
    "_catalog_declares_kernel",
    "_operand_kernel_axis",
    "_require_cataloged_kernel",
)
