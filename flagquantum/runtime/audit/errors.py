"""Exceptions raised by distributed audit and release gates."""

from __future__ import annotations

from .schema import DistributedScalabilityAudit


class DistributedScalabilityError(AssertionError):
    """Raised when a payload fails FlagQuantum scalability requirements."""

    def __init__(self, audit: DistributedScalabilityAudit) -> None:
        self.audit = audit
        details = "; ".join(
            audit.errors or audit.warnings or ("scalability audit failed",)
        )
        super().__init__(details)


__all__ = ("DistributedScalabilityError",)
