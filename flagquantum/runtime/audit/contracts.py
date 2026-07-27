"""Audit-owned validation of typed execution records."""

from __future__ import annotations

from collections.abc import Sequence

from ...core.contracts import (
    AuditRecordContract,
    AuditValidation,
    ExecutionRecordContract,
)


def audit_execution_record(
    record: ExecutionRecordContract,
    *,
    validator: str,
    errors: Sequence[str] = (),
    require_sharded: bool = True,
) -> AuditRecordContract:
    """Validate an observed record and exclusively own release approval."""

    found = list(errors)
    if not record.observed.completed:
        found.append("execution_not_completed")
    if record.failures:
        found.append("execution_failures_present")
    if (
        require_sharded
        and record.ownership.distribution_semantics != "sharded_across_ranks"
    ):
        found.append("release_requires_sharded_across_ranks")
    if not record.provenance.commit or not record.provenance.workload_sha256:
        found.append("production_provenance_incomplete")
    unique_errors = tuple(dict.fromkeys(found))
    approved = not unique_errors
    return AuditRecordContract(
        execution_hash=record.content_hash,
        validated=AuditValidation(
            validator=validator,
            passed=approved,
            release_claim_approved=approved,
            errors=unique_errors,
        ),
    )


__all__ = ["audit_execution_record"]
