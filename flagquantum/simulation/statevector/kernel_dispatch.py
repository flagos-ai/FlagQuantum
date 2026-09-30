"""Shared fail-closed authorization for cataloged statevector kernels."""

from __future__ import annotations

from ...kernels.catalog import KernelImplementation, KernelMatchResult


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


__all__ = ("_require_cataloged_kernel",)
