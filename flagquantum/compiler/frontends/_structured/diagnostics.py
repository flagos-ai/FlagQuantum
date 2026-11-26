"""Shared diagnostics for static specialization and dynamic lowering."""

from __future__ import annotations

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.model import SourceLocation


class SpecializationError(ValueError):
    """Raised when runtime values cannot specialize the accepted IR profile."""

    def __init__(
        self, code: str, message: str, location: SourceLocation | None = None
    ) -> None:
        self.code = code
        self.location = location
        prefix = "" if location is None else f"{location.filename}:{location.line}: "
        super().__init__(f"{prefix}{code}: {message}")


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (SpecializationError,):
    _identity_export.__module__ = "flagquantum.compiler._hybrid.diagnostics"
del _identity_export

_alias_module("flagquantum.compiler._hybrid.diagnostics", __name__)
