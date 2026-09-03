"""Characterization tests for the Runtime-owned execution-attempt lifecycle."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.contracts import (
    ExecutionObservation,
    FailureContract,
    MeasurementContract,
    OwnershipContract,
    ProvenanceContract,
    RuntimePlanContract,
)
from flagquantum.errors import ExecutionError
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
)
from flagquantum.runtime.records import record_execution
from flagquantum.runtime.routing import FallbackEvent, StrictExecutionScope
from flagquantum.runtime.trajectories.checkpoint import (
    TrajectoryCheckpoint,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
)
from flagquantum.runtime.trajectories.result import TrajectoryFailure
from flagquantum.runtime.trajectories.statistics import TensorWelford


def _bell() -> fq.Circuit:
    return fq.Circuit(2).h(0).cx(0, 1)


@pytest.mark.integration
def test_validated_plan_executes_once_without_replanning_or_recompiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(_bell(), options=fq.ExecutionOptions(mode="statevector"))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an execution attempt must consume its supplied plan")

    monkeypatch.setattr("flagquantum.compilation.planner.plan", forbidden)
    monkeypatch.setattr("flagquantum.runtime.execution.compile_for_backend", forbidden)

    result = fq.run(plan)
    summary = result.summary()

    assert result.plan is plan
    assert summary["schema"] == "flagquantum.execution_result.summary"
    assert summary["has_plan"] is True
    assert summary["has_state"] is True
    assert summary["runtime"]["mode"] == "statevector"


@pytest.mark.integration
def test_attempt_failure_is_normalized_once_and_retains_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(_bell(), options=fq.ExecutionOptions(mode="statevector"))
    launches = 0

    def fail_once(*args: object, **kwargs: object) -> None:
        nonlocal launches
        launches += 1
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", fail_once)

    with pytest.raises(ExecutionError, match="planned execution failed") as captured:
        fq.run(plan)

    assert launches == 1
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "kernel launch failed"


@pytest.mark.unit
def test_fallback_is_explicit_and_propagates_into_runtime_evidence() -> None:
    event = FallbackEvent(
        operator="aten::linalg_svd",
        source_route="device_native",
        target_route="host",
        reason="debug-only numerical comparison",
        host_transfer=True,
    )
    scope = StrictExecutionScope("host_debug_only")
    scope.record_fallback(event)

    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.DEVELOPMENT_RUN,
        evidence_scope=EvidenceScope.ONE_GPU_LOCAL,
        provenance=RuntimeProvenance(
            commit="a" * 40,
            workload_sha256="b" * 64,
            command=("python", "attempt.py"),
            devices=("cpu",),
            topology="single_process",
            rank_mapping=("rank0=cpu",),
            collective_backend="none",
            warmup=0,
            iterations=1,
            seeds=(7,),
            raw_log_sha256="c" * 64,
            fallback_events=(event.reason,),
        ),
        evidence={"status": "completed_with_debug_fallback"},
        signing_key=b"runtime-characterization-key",
    )

    assert scope.fallback_events == (event,)
    assert scope.production_eligible is False
    assert artifact.summary()["provenance"]["fallback_events"] == (event.reason,)


@pytest.mark.unit
def test_trajectory_checkpoint_restores_identity_progress_and_retry_work(
    tmp_path,
) -> None:
    accumulator = TensorWelford()
    accumulator.update(torch.tensor([0.25]))
    accumulator.update(torch.tensor([0.75]))
    checkpoint = TrajectoryCheckpoint(
        requested_count=5,
        base_seed=19,
        completed_ids=(0, 2),
        failures=(
            TrajectoryFailure(
                trajectory_id=1,
                error_type="TimeoutError",
                message="rank-local timeout",
                retryable=True,
            ),
        ),
        statistics_state=accumulator.state_dict(),
        noise_model_identity="noise:sha256",
        circuit_digest="circuit:sha256",
        execution_metadata={"distribution_semantics": "replicated_per_rank"},
    )

    restored = load_trajectory_checkpoint(
        save_trajectory_checkpoint(checkpoint, tmp_path / "trajectory.pt")
    )

    assert restored.to_payload()["version"] == checkpoint.version
    assert restored.completed_ids == (0, 2)
    assert restored.pending_ids(retry_failed=True) == (1, 3, 4)
    assert restored.pending_ids(retry_failed=False) == (3, 4)
    assert restored.accumulator().count == 2
    assert restored.execution_metadata == {
        "distribution_semantics": "replicated_per_rank"
    }


@pytest.mark.unit
def test_attempt_record_preserves_success_and_failure_evidence() -> None:
    plan = RuntimePlanContract(plan_id="plan-001")
    ownership = OwnershipContract(
        rank_owners=(("state", 0),),
        distribution_semantics="single_device_fast_path",
    )
    provenance = ProvenanceContract(
        commit="a" * 40,
        workload_sha256="b" * 64,
        command=("python", "attempt.py"),
        devices=("cpu",),
        rank_mapping=("rank0=cpu",),
        raw_log_sha256="c" * 64,
    )
    success = record_execution(
        plan,
        observed=ExecutionObservation(
            executor="pytorch_statevector",
            elapsed_seconds=0.01,
            peak_memory_bytes=256,
            communication_bytes=0,
            completed=True,
        ),
        ownership=ownership,
        provenance=provenance,
        measurements=(
            MeasurementContract(kind_name="expectation", wires=(0,), values=(1.0,)),
        ),
    )
    failure = record_execution(
        plan,
        observed=ExecutionObservation(
            executor="pytorch_statevector",
            elapsed_seconds=0.02,
            peak_memory_bytes=256,
            communication_bytes=0,
            completed=False,
        ),
        ownership=ownership,
        provenance=provenance,
        failures=(
            FailureContract(
                code="kernel_launch_failed",
                message="kernel launch failed",
                retryable=True,
                blockers=("device_unhealthy",),
            ),
        ),
    )

    assert success.plan_id == failure.plan_id == plan.plan_id
    assert success.observed.completed is True
    assert success.measurements[0].values == (1.0,)
    assert success.failures == ()
    assert failure.observed.completed is False
    assert failure.failures[0].retryable is True
    assert failure.failures[0].blockers == ("device_unhealthy",)
