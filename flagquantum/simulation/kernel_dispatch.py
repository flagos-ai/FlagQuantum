"""Shared fail-closed authorization for cataloged simulation kernels."""

from __future__ import annotations

import os

from ..kernels.catalog import KernelImplementation, KernelMatchResult

_DISABLED_FLAG_VALUES = frozenset({"0", "false", "off", "no"})


def _opt_in_environment_flag(name: str, default: str = "0") -> bool:
    """Return whether environment variable ``name`` enables its route.

    The four spellings that mean "off" are stated here once, because two
    routes that parsed their own variable could disagree about what "off"
    means while both reporting the same metadata field.
    """

    return os.getenv(name, default).strip().lower() not in _DISABLED_FLAG_VALUES


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


__all__ = ("_opt_in_environment_flag", "_require_cataloged_kernel")
