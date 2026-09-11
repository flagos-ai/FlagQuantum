"""Executor-owned construction of observed runtime records."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.contracts import (
    AccuracyContract,
    ExecutionObservation,
    ExecutionRecordContract,
    FailureContract,
    MeasurementContract,
    OwnershipContract,
    ProvenanceContract,
    RuntimePlanContract,
)


def record_execution(
    plan: RuntimePlanContract,
    *,
    observed: ExecutionObservation,
    ownership: OwnershipContract,
    provenance: ProvenanceContract,
    measurements: Sequence[MeasurementContract] = (),
    accuracy: AccuracyContract | None = None,
    failures: Sequence[FailureContract] = (),
) -> ExecutionRecordContract:
    """Build an executor record; release approval is intentionally unavailable."""

    return ExecutionRecordContract(
        plan_id=plan.plan_id,
        observed=observed,
        ownership=ownership,
        measurements=tuple(measurements),
        accuracy=accuracy or AccuracyContract(),
        failures=tuple(failures),
        provenance=provenance,
    )


__all__ = ["record_execution"]
