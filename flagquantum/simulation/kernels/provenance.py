"""Kernel compiler provenance resolved from installed package metadata."""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata

from flagquantum._import_compat import alias_module as _alias_module


@lru_cache(maxsize=1)
def triton_compiler_provenance() -> tuple[str | None, str | None, str, str]:
    """Resolve which distribution installed ``triton`` without importing it."""

    module_owners = metadata.packages_distributions().get("triton", ())
    owners = tuple(
        sorted({owner.casefold().replace("_", "-") for owner in module_owners})
    )
    if not owners:
        return None, None, "unknown", "missing"
    if len(owners) != 1:
        return None, None, "unknown", "ambiguous"

    distribution = owners[0]
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return distribution, None, "unknown", "missing_metadata"

    if distribution == "triton":
        return distribution, version, "direct", "resolved"
    if distribution == "flagtree":
        return distribution, version, "flagtree", "resolved"
    return distribution, version, "unknown", "unsupported_distribution"


__all__ = ("triton_compiler_provenance",)


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (triton_compiler_provenance,):
    _identity_export.__module__ = "flagquantum.kernels.provenance"
del _identity_export

_alias_module("flagquantum.kernels.provenance", __name__)
